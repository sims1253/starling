#include "engine_gguf.hpp"
#include "engine_math.hpp"
#include "engine_encoder.hpp"
#include "engine_decoder.hpp"
#include "engine_audio.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
uint32_t u32(const uint8_t* p) {
    return uint32_t(p[0]) | uint32_t(p[1])<<8 | uint32_t(p[2])<<16 | uint32_t(p[3])<<24;
}
uint16_t u16(const uint8_t* p) { return uint16_t(p[0]) | uint16_t(p[1])<<8; }
std::vector<float> wav(const std::string& path) {
    std::ifstream f(path,std::ios::binary);
    if(!f) throw std::runtime_error("cannot open WAV");
    f.seekg(0,std::ios::end); size_t size=size_t(f.tellg()); f.seekg(0);
    std::vector<uint8_t> bytes(size); f.read(reinterpret_cast<char*>(bytes.data()),size);
    if(size<44 || std::memcmp(bytes.data(),"RIFF",4) || std::memcmp(bytes.data()+8,"WAVE",4))
        throw std::runtime_error("invalid WAV");
    size_t pos=12; const uint8_t* data=nullptr; size_t data_size=0;
    bool fmt=false;
    while(pos+8<=size) {
        uint32_t n=u32(bytes.data()+pos+4);
        if(pos+8+n>size) throw std::runtime_error("truncated WAV chunk");
        if(!std::memcmp(bytes.data()+pos,"fmt ",4)) {
            const uint8_t* p=bytes.data()+pos+8;
            if(n<16 || u16(p)!=1 || u16(p+2)!=1 || u32(p+4)!=16000 || u16(p+14)!=16)
                throw std::runtime_error("expected 16 kHz mono PCM16 WAV");
            fmt=true;
        } else if(!std::memcmp(bytes.data()+pos,"data",4)) {
            data=bytes.data()+pos+8;data_size=n;
        }
        pos+=8+n+(n&1);
    }
    if(!fmt||!data||data_size%2) throw std::runtime_error("missing WAV format or data");
    std::vector<float> pcm(data_size/2);
    for(size_t i=0;i<pcm.size();++i) pcm[i]=float(int16_t(u16(data+2*i)))/32768.0f;
    return pcm;
}
}

int main(int argc,char** argv) {
    try {
        if(argc<3) {
            std::cerr<<"usage: standalone_parakeet MODEL.gguf AUDIO.wav [threads] "
                        "[--pack-q4] [--iterations N] [--dump-encoder FILE] "
                        "[--input-encoder FILE] [--benchmark-window-s S --benchmark-advance-s S]\n";
            return 2;
        }
        int threads=4, iterations=1; bool pack=false; std::string dump,dump_mel,input_encoder;
        double stream_window_s=0,stream_advance_s=0;
        double benchmark_window_s=0,benchmark_advance_s=0;
        for(int i=3;i<argc;++i) {
            if(std::string(argv[i])=="--pack-q4") pack=true;
            else if(std::string(argv[i])=="--dump-encoder" && i+1<argc) dump=argv[++i];
            else if(std::string(argv[i])=="--input-encoder" && i+1<argc) input_encoder=argv[++i];
            else if(std::string(argv[i])=="--dump-mel" && i+1<argc) dump_mel=argv[++i];
            else if(std::string(argv[i])=="--stream-window-s" && i+1<argc) stream_window_s=std::stod(argv[++i]);
            else if(std::string(argv[i])=="--stream-advance-s" && i+1<argc) stream_advance_s=std::stod(argv[++i]);
            else if(std::string(argv[i])=="--benchmark-window-s" && i+1<argc) benchmark_window_s=std::stod(argv[++i]);
            else if(std::string(argv[i])=="--benchmark-advance-s" && i+1<argc) benchmark_advance_s=std::stod(argv[++i]);
            else if(std::string(argv[i])=="--iterations" && i+1<argc) iterations=std::stoi(argv[++i]);
            else threads=std::stoi(argv[i]);
        }
        if(threads<1 || iterations<1) throw std::runtime_error("thread count and iterations must be positive");
        const auto start=std::chrono::steady_clock::now();
        direct::Model model(argv[1]);
        direct::Kernels kernels(threads,pack);
        direct::Encoder encoder(model,kernels);
        direct::Decoder decoder(model,kernels);
        auto pcm=wav(argv[2]);
        if(benchmark_window_s>0) {
            const size_t window=size_t(std::llround(benchmark_window_s*16000));
            const size_t advance=size_t(std::llround(benchmark_advance_s*16000));
            if(window==0 || advance==0 || advance>window || stream_window_s>0)
                throw std::runtime_error("invalid benchmark window or advance");
            std::vector<std::pair<size_t,size_t>> windows;
            size_t first=0;
            while(first+window<=pcm.size()) {
                windows.emplace_back(first,window);
                first+=advance;
            }
            if(first<pcm.size()) windows.emplace_back(first,pcm.size()-first);
            if(windows.empty()) throw std::runtime_error("empty benchmark audio");
            int warm_frames=0;
            auto warm=encoder.run(pcm.data()+windows[0].first,windows[0].second,warm_frames);
            (void)decoder.decode(warm,warm_frames);
            std::vector<double> times;
            std::vector<double> cpu_times;
            std::vector<std::string> reference_texts;
            struct WindowTiming { double encoder,decoder,total; };
            std::vector<WindowTiming> window_times;
            for(int run=0;run<iterations;++run) {
                encoder.reset_stream();
                const auto begin=std::chrono::steady_clock::now();
                const auto cpu_begin=std::clock();
                for(size_t i=0;i<windows.size();++i) {
                    const auto [offset,count]=windows[i];
                    const auto window_start=std::chrono::steady_clock::now();
                    int frames=0;
                    auto encoded=encoder.run_window(pcm.data()+offset,count,offset,frames);
                    const auto encoded_at=std::chrono::steady_clock::now();
                    auto ids=decoder.decode(encoded,frames);
                    const auto decoded_at=std::chrono::steady_clock::now();
                    auto transcript=decoder.text(ids);
                    if(run==0) {
                        reference_texts.push_back(transcript);
                        auto sec=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};
                        window_times.push_back({sec(window_start,encoded_at),
                                                sec(encoded_at,decoded_at),
                                                sec(window_start,decoded_at)});
                    } else if(transcript!=reference_texts[i]) {
                        throw std::runtime_error("stream benchmark transcription changed between runs");
                    }
                }
                times.push_back(std::chrono::duration<double>(
                    std::chrono::steady_clock::now()-begin).count());
                cpu_times.push_back(double(std::clock()-cpu_begin)/CLOCKS_PER_SEC);
            }
            auto sorted=times;
            std::sort(sorted.begin(),sorted.end());
            const double median=(sorted[(sorted.size()-1)/2]+sorted[sorted.size()/2])*0.5;
            for(size_t i=0;i<windows.size();++i) {
                std::cout<<"window="<<i<<" first_sample="<<windows[i].first
                         <<" samples="<<windows[i].second
                         <<" encoder_s="<<window_times[i].encoder
                         <<" decoder_s="<<window_times[i].decoder
                         <<" total_s="<<window_times[i].total
                         <<" transcript="<<reference_texts[i]<<'\n';
            }
            std::cout<<"backend=direct-cpu audio_s="<<double(pcm.size())/16000
                     <<" threads="<<threads<<" windows="<<windows.size()
                     <<" benchmark_window_s="<<benchmark_window_s
                     <<" benchmark_advance_s="<<benchmark_advance_s
                     <<" session_median_s="<<median<<" session_times_s=";
            for(size_t i=0;i<times.size();++i) std::cout<<(i?",":"")<<times[i];
            std::cout<<" session_cpu_s=";
            for(size_t i=0;i<cpu_times.size();++i) std::cout<<(i?",":"")<<cpu_times[i];
            std::cout<<'\n';
            return 0;
        }
        if(stream_window_s>0) {
            const size_t window=size_t(std::llround(stream_window_s*16000));
            const size_t advance=size_t(std::llround(stream_advance_s*16000));
            if(advance==0 || window==0 || advance+window>pcm.size())
                throw std::runtime_error("two full streaming windows must fit in WAV");
            direct::MelCache cache;
            (void)direct::compute_mel(model,pcm.data(),window,&cache,0);
            const auto mel_start=std::chrono::steady_clock::now();
            auto cached_mel=direct::compute_mel(model,pcm.data()+advance,window,&cache,advance);
            const auto mel_cached_end=std::chrono::steady_clock::now();
            const size_t hits=cache.reused_frames;
            auto fresh_mel=direct::compute_mel(model,pcm.data()+advance,window);
            const auto mel_fresh_end=std::chrono::steady_clock::now();
            bool mel_exact=cached_mel.frames==fresh_mel.frames &&
                cached_mel.values==fresh_mel.values;
            std::vector<float> changed(pcm.begin()+advance,pcm.begin()+advance+window);
            changed[window/2]+=0.125f;
            auto changed_cached=direct::compute_mel(model,changed.data(),window,&cache,advance);
            const size_t changed_hits=cache.reused_frames;
            auto changed_fresh=direct::compute_mel(model,changed.data(),window);
            bool changed_exact=changed_cached.values==changed_fresh.values;
            int t0=0,t1=0,tf=0;
            auto e0=encoder.run_window(pcm.data(),window,0,t0);
            (void)e0;
            auto e1=encoder.run_window(pcm.data()+advance,window,advance,t1);
            const size_t encoder_hits=encoder.last_reused_mel_frames();
            direct::Encoder fresh_encoder(model,kernels);
            auto ef=fresh_encoder.run(pcm.data()+advance,window,tf);
            bool encoder_exact=t1==tf && e1==ef;
            auto ids=decoder.decode(e1,t1);
            auto fresh_ids=decoder.decode(ef,tf);
            std::cout<<"backend=direct-cpu stream_window_s="<<stream_window_s
                     <<" stream_advance_s="<<stream_advance_s
                     <<" reused_mel_frames="<<hits<<" encoder_reused_mel_frames="<<encoder_hits
                     <<" total_mel_frames="<<cached_mel.frames
                     <<" mel_exact="<<mel_exact<<" encoder_exact="<<encoder_exact
                     <<" ids_exact="<<(ids==fresh_ids)
                     <<" changed_pcm_reused_frames="<<changed_hits
                     <<" changed_pcm_exact="<<changed_exact<<'\n';
            std::cout<<"mel_cached_s="<<std::chrono::duration<double>(mel_cached_end-mel_start).count()
                     <<" mel_fresh_s="<<std::chrono::duration<double>(mel_fresh_end-mel_cached_end).count()<<'\n';
            std::cout<<"transcript="<<decoder.text(ids)<<'\n';
            if(!mel_exact || !encoder_exact || ids!=fresh_ids ||
               changed_hits!=0 || !changed_exact) return 1;
            return 0;
        }
        if(!dump_mel.empty()) {
            auto mel=direct::compute_mel(model,pcm.data(),pcm.size());
            std::ofstream f(dump_mel,std::ios::binary);
            f.write(reinterpret_cast<const char*>(&mel.frames),sizeof(mel.frames));
            f.write(reinterpret_cast<const char*>(mel.values.data()),std::streamsize(mel.values.size()*sizeof(float)));
        }
        const auto ready=std::chrono::steady_clock::now();
        int frames=0;
        direct::Buffer enc;
        std::vector<int32_t> ids;
        double encoder_s=0,decoder_s=0;
        for(int run=0;run<iterations;++run) {
            const auto run_start=std::chrono::steady_clock::now();
            if(input_encoder.empty()) enc=encoder.run(pcm.data(),pcm.size(),frames);
            else {
                std::ifstream f(input_encoder,std::ios::binary);
                if(!f.read(reinterpret_cast<char*>(&frames),sizeof(frames)) || frames<1)
                    throw std::runtime_error("invalid input encoder header");
                const int width=int(model.at("joint.enc.weight").rows());
                enc.resize(size_t(frames)*width);
                if(!f.read(reinterpret_cast<char*>(enc.data()),
                           std::streamsize(enc.size()*sizeof(float))))
                    throw std::runtime_error("invalid input encoder data");
            }
            const auto encoded=std::chrono::steady_clock::now();
            ids=decoder.decode(enc,frames);
            const auto decoded=std::chrono::steady_clock::now();
            auto sec=[](auto a,auto b) {return std::chrono::duration<double>(b-a).count();};
            encoder_s=sec(run_start,encoded);decoder_s=sec(encoded,decoded);
            std::cerr<<"[DIRECT_TIMING] run="<<run<<" encoder_s="<<encoder_s
                     <<" decoder_s="<<decoder_s<<" total_s="<<sec(run_start,decoded)<<'\n';
        }
        if(!dump.empty()) {
            std::ofstream f(dump,std::ios::binary);
            f.write(reinterpret_cast<const char*>(&frames),sizeof(frames));
            f.write(reinterpret_cast<const char*>(enc.data()),std::streamsize(enc.size()*sizeof(float)));
        }
        auto secs=[](auto a,auto b) {return std::chrono::duration<double>(b-a).count();};
        std::cout<<"backend=direct-cpu threads="<<threads<<" pack_q4="<<pack
                 <<" audio_s="<<double(pcm.size())/16000<<" frames="<<frames
                 <<" load_s="<<secs(start,ready)<<" encoder_s="<<encoder_s
                 <<" decoder_s="<<decoder_s<<" total_s="<<encoder_s+decoder_s<<'\n';
        std::cout<<"transcript="<<decoder.text(ids)<<'\n';
        const auto& by_type=kernels.seconds_by_type();
        std::cout<<"matmul_s f32="<<by_type[0]<<" f16="<<by_type[1]
                 <<" q8_0="<<by_type[8]<<" q4_k="<<by_type[12]
                 <<" q6_k="<<by_type[14]<<'\n';
        const auto& stages=encoder.stage_seconds();
        std::cout<<"stage_s mel="<<stages[0]<<" subsample="<<stages[1]
                 <<" positions="<<stages[2]<<" ff1="<<stages[3]
                 <<" attention="<<stages[4]<<" conv="<<stages[5]
                 <<" ff2="<<stages[6]<<" norm_out="<<stages[7]
                 <<" joint_enc="<<stages[8]<<" silu="<<stages[9]<<'\n';
        std::cout<<"ids=";
        for(size_t i=0;i<ids.size();++i) std::cout<<(i?",":"")<<ids[i];
        std::cout<<'\n';
    } catch(const std::exception& e) {
        std::cerr<<"error: "<<e.what()<<'\n'; return 1;
    }
}
