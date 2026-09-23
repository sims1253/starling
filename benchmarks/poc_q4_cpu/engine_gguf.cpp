#include "engine_gguf.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if defined(__F16C__)
#include <immintrin.h>
#endif

namespace direct {
namespace {
uint64_t number(std::ifstream& f, int width) {
    uint64_t v = 0;
    for (int i = 0; i < width; ++i) {
        int c = f.get();
        if (c == EOF) throw std::runtime_error("truncated GGUF metadata");
        v |= uint64_t(uint8_t(c)) << (8*i);
    }
    return v;
}
std::string str(std::ifstream& f) {
    const uint64_t n = number(f,8);
    if (n > (1u << 26)) throw std::runtime_error("GGUF string too long");
    std::string s(size_t(n), '\0');
    f.read(s.data(), std::streamsize(n));
    if (!f) throw std::runtime_error("truncated GGUF string");
    return s;
}
int width(uint32_t t) {
    switch(t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: throw std::runtime_error("unsupported GGUF metadata type");
    }
}
void skip(std::ifstream& f, uint64_t n) {
    if (n > uint64_t(std::numeric_limits<std::streamoff>::max()))
        throw std::runtime_error("GGUF metadata overflow");
    f.seekg(std::streamoff(n), std::ios::cur);
    if (!f) throw std::runtime_error("truncated GGUF metadata");
}
int64_t signed_value(uint64_t v, int bytes) {
    if (bytes == 1) return int8_t(v);
    if (bytes == 2) return int16_t(v);
    if (bytes == 4) return int32_t(v);
    return int64_t(v);
}
uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1])<<8); }
size_t tensor_bytes(const Tensor& t) {
    int64_t n = 1;
    for (int d=0;d<t.rank;++d) {
        if(t.shape[d]<=0 || n>INT64_MAX/t.shape[d])
            throw std::runtime_error("invalid tensor shape");
        n *= t.shape[d];
    }
    if(uint64_t(n)>SIZE_MAX/4) throw std::runtime_error("tensor size overflow");
    switch(t.type) {
        case 0: return size_t(n)*4;
        case 1: return size_t(n)*2;
        case 8: if(n%32) break; return size_t(n/32)*34;
        case 12: if(n%256) break; return size_t(n/256)*144;
        case 14: if(n%256) break; return size_t(n/256)*210;
    }
    throw std::runtime_error("unsupported tensor type or shape");
}
} // namespace

float half(uint16_t bits) {
#if defined(__F16C__)
    return _cvtsh_ss(bits);
#else
    const int e=(bits>>10)&31, m=bits&1023;
    float x=e==0?std::ldexp(float(m),-24):e==31?(m?NAN:INFINITY):std::ldexp(float(1024+m),e-25);
    return bits&0x8000 ? -x:x;
#endif
}

Model::Model(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open model: "+path);
    char magic[4]; f.read(magic,4);
    if (!f || std::memcmp(magic,"GGUF",4) || number(f,4)!=3)
        throw std::runtime_error("GGUF v3 required");
    const uint64_t nt=number(f,8), nk=number(f,8);
    if(nt>1000000 || nk>1000000) throw std::runtime_error("GGUF metadata count too large");
    uint64_t align=32;
    for(uint64_t i=0;i<nk;++i) {
        const std::string key=str(f);
        const uint32_t t=uint32_t(number(f,4));
        if(t==9) {
            const uint32_t item=uint32_t(number(f,4));
            const uint64_t n=number(f,8);
            if(n>10000000) throw std::runtime_error("GGUF array too large");
            if(item==8) {
                std::vector<std::string> a;
                if(key=="parakeet.tokenizer.pieces") a.reserve(size_t(n));
                for(uint64_t j=0;j<n;++j) {
                    auto s=str(f);
                    if(key=="parakeet.tokenizer.pieces") a.push_back(std::move(s));
                }
                if(!a.empty()) string_arrays_.emplace(key,std::move(a));
            } else {
                const int w=width(item);
                if(key=="parakeet.tdt.durations") {
                    auto& a=int_arrays_[key]; a.reserve(size_t(n));
                    for(uint64_t j=0;j<n;++j) a.push_back(signed_value(number(f,w),w));
                } else skip(f,n*uint64_t(w));
            }
        } else if(t==8) { (void)str(f); }
        else if(t==6) {
            uint32_t u=uint32_t(number(f,4)); float v; std::memcpy(&v,&u,4);
            floats_[key]=v;
        } else if(t==12) {
            uint64_t u=number(f,8); double v; std::memcpy(&v,&u,8);
            floats_[key]=v;
        } else {
            const int w=width(t); const uint64_t v=number(f,w);
            ints_[key]=(t==1||t==3||t==5||t==11)?signed_value(v,w):int64_t(v);
            if(key=="general.alignment") align=v;
        }
    }
    if(!align || (align&(align-1)) || align>4096) throw std::runtime_error("invalid alignment");
    for(uint64_t i=0;i<nt;++i) {
        std::string name=str(f); Tensor t; t.rank=int(number(f,4));
        if(t.rank<1 || t.rank>4) throw std::runtime_error("invalid tensor rank");
        for(int d=0;d<t.rank;++d) t.shape[d]=int64_t(number(f,8));
        t.type=uint32_t(number(f,4)); t.offset=number(f,8);
        if(!tensors_.emplace(name,t).second) throw std::runtime_error("duplicate tensor: "+name);
    }
    const uint64_t data_start=(uint64_t(f.tellg())+align-1)&~(align-1);
    int fd=open(path.c_str(),O_RDONLY);
    if(fd<0) throw std::runtime_error("cannot mmap model");
    struct stat st{};
    if(fstat(fd,&st)) { close(fd); throw std::runtime_error("cannot stat model"); }
    file_size_=size_t(st.st_size);
    if(data_start>file_size_) { close(fd); throw std::runtime_error("truncated GGUF data section"); }
    for(const auto& [name,t]:tensors_) {
        const size_t n=tensor_bytes(t);
        if(t.offset>file_size_-data_start || n>file_size_-data_start-t.offset) {
            close(fd);
            throw std::runtime_error("truncated tensor: "+name);
        }
    }
    mapping_=mmap(nullptr,file_size_,PROT_READ,MAP_PRIVATE,fd,0);
    close(fd);
    if(mapping_==MAP_FAILED) { mapping_=nullptr; throw std::runtime_error("mmap failed"); }
    for(auto& [name,t]:tensors_) {
        (void)name;
        t.data=static_cast<const uint8_t*>(mapping_)+data_start+t.offset;
    }
}
Model::~Model() { if(mapping_) munmap(mapping_,file_size_); }
const Tensor& Model::at(const std::string& n) const {
    auto it=tensors_.find(n);
    if(it==tensors_.end()) throw std::runtime_error("missing tensor: "+n);
    return it->second;
}
const Tensor* Model::find(const std::string& n) const {
    auto it=tensors_.find(n); return it==tensors_.end()?nullptr:&it->second;
}
int64_t Model::integer(const std::string& n,int64_t d) const { auto i=ints_.find(n); return i==ints_.end()?d:i->second; }
double Model::real(const std::string& n,double d) const {
    auto f=floats_.find(n); if(f!=floats_.end()) return f->second;
    auto i=ints_.find(n); return i==ints_.end()?d:double(i->second);
}
std::vector<int64_t> Model::integers(const std::string& n) const { auto i=int_arrays_.find(n); return i==int_arrays_.end()?std::vector<int64_t>{}:i->second; }
std::vector<std::string> Model::strings(const std::string& n) const { auto i=string_arrays_.find(n); return i==string_arrays_.end()?std::vector<std::string>{}:i->second; }

float scalar(const Tensor& t,size_t i) {
    if(t.type==0) { float x; std::memcpy(&x,t.data+i*4,4); return x; }
    if(t.type==1) return half(u16(t.data+i*2));
    throw std::runtime_error("tensor is not scalar float");
}
void dequant_row(const Tensor& t,int64_t row,float* out) {
    const int64_t cols=t.cols();
    if(t.type==0) { std::memcpy(out,t.data+size_t(row*cols)*4,size_t(cols)*4); return; }
    if(t.type==1) { for(int64_t i=0;i<cols;++i) out[i]=half(u16(t.data+size_t(row*cols+i)*2)); return; }
    if(t.type==8) {
        for(int64_t b=0;b<cols/32;++b) {
            const uint8_t* p=t.data+size_t(row*(cols/32)+b)*34;
            float d=half(u16(p));
            for(int i=0;i<32;++i) out[b*32+i]=d*int8_t(p[2+i]);
        }
        return;
    }
    if(t.type==14) {
        for(int64_t b=0;b<cols/256;++b) {
            const uint8_t* p=t.data+size_t(row*(cols/256)+b)*210;
            const uint8_t *ql=p,*qh=p+128;
            const int8_t* sc=reinterpret_cast<const int8_t*>(p+192);
            float d=half(u16(p+208));
            for(int n=0;n<256;n+=128) for(int l=0;l<32;++l) {
                int off=n/128*64, hi=n/128*32, si=n/128*8+l/16;
                out[b*256+n+l]    =d*sc[si+0]*(int((ql[off+l]&15)|(((qh[hi+l]>>0)&3)<<4))-32);
                out[b*256+n+l+32] =d*sc[si+2]*(int((ql[off+l+32]&15)|(((qh[hi+l]>>2)&3)<<4))-32);
                out[b*256+n+l+64] =d*sc[si+4]*(int((ql[off+l]>>4)|(((qh[hi+l]>>4)&3)<<4))-32);
                out[b*256+n+l+96] =d*sc[si+6]*(int((ql[off+l+32]>>4)|(((qh[hi+l]>>6)&3)<<4))-32);
            }
        }
        return;
    }
    throw std::runtime_error("unsupported dequant tensor");
}
} // namespace direct
