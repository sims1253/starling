#include "golden_io.hpp"
#include <cstdio>
#include <fstream>
#include <regex>
namespace starling::ggml::test {
template<class T> bool raw(const std::string&p,std::vector<T>&v,std::string&e){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f){e="open failed: "+p;return false;}auto n=f.tellg();if(n<0||n%(std::streamoff)sizeof(T)){e="invalid raw size: "+p;return false;}v.resize((size_t)n/sizeof(T));f.seekg(0);return !!f.read((char*)v.data(),n);}
bool read_f32(const std::string&p,std::vector<float>&v,std::string&e){return raw(p,v,e);} bool read_i64(const std::string&p,std::vector<int64_t>&v,std::string&e){return raw(p,v,e);}
bool read_shape(const std::string&p,std::vector<int64_t>&v,std::string&e){std::ifstream f(p);if(!f){e="open failed: "+p;return false;}std::string s((std::istreambuf_iterator<char>(f)),{});auto a=s.find('['),b=s.find(']',a);if(a==std::string::npos||b==std::string::npos){e="missing JSON shape: "+p;return false;}std::regex r("[0-9]+");for(std::sregex_iterator i(s.begin()+a,s.begin()+b,r),z;i!=z;++i)v.push_back(std::stoll(i->str()));return !v.empty();}
}
