#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

volatile float sink=0;
#define NI __attribute__((noinline))
#if defined(__GNUC__) && !defined(__clang__)
#define NOIF __attribute__((optimize("no-if-conversion,no-if-conversion2")))
#else
#define NOIF
#endif

// Source-faithful PyTorch NCHW window scan: -inf then all four values.
extern "C" NI NOIF float branched_maxpool4(const float* x){
 float m=-std::numeric_limits<float>::infinity();
#pragma GCC unroll 0
 for(int i=0;i<4;++i) if(x[i]>m) m=x[i];
 return m;
}
extern "C" NI float control_load4(const float* x){
 float s=0;
#pragma GCC unroll 0
 for(int i=0;i<4;++i) s+=x[i];
 return s;
}

struct Counts{uint64_t branches=0,misses=0;};
#ifdef __linux__
int open_event(uint64_t config,int group){perf_event_attr a{};a.type=PERF_TYPE_HARDWARE;a.size=sizeof(a);a.config=config;a.disabled=group==-1;a.exclude_kernel=1;a.exclude_hv=1;return syscall(SYS_perf_event_open,&a,0,-1,group,0);}
class PMU{int b=-1,m=-1;public:PMU(){b=open_event(PERF_COUNT_HW_BRANCH_INSTRUCTIONS,-1);if(b<0)throw std::runtime_error("perf_event_open branches: "+std::string(strerror(errno)));m=open_event(PERF_COUNT_HW_BRANCH_MISSES,b);if(m<0)throw std::runtime_error("perf_event_open misses: "+std::string(strerror(errno)));}~PMU(){if(m>=0)close(m);if(b>=0)close(b);}void start(){ioctl(b,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP);if(ioctl(b,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP)<0)throw std::runtime_error("PMU enable failed");}Counts stop(){ioctl(b,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP);Counts c;if(read(b,&c.branches,8)!=8||read(m,&c.misses,8)!=8)throw std::runtime_error("PMU read failed");return c;}};
#else
class PMU{public:PMU(){throw std::runtime_error("Linux PMU required");}void start(){}Counts stop(){return {};}};
#endif

std::vector<float> load(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("cannot open input");auto n=f.tellg();if(n<=0||n%16)throw std::runtime_error("input must be four-float windows");f.seekg(0);std::vector<float>x(size_t(n)/4);f.read(reinterpret_cast<char*>(x.data()),n);return x;}
float replay(const std::vector<float>&x,uint64_t repeats,bool branch){float sum=0;for(uint64_t r=0;r<repeats;++r)for(size_t i=0;i<x.size();i+=4)sum+=branch?branched_maxpool4(x.data()+i):control_load4(x.data()+i);sink=sum;return sum;}

int main(int argc,char**argv){try{std::string path,mode="branched";uint64_t repeats=100,warmup=5;for(int i=1;i<argc;++i){std::string s=argv[i];if(s=="--input")path=argv[++i];else if(s=="--mode")mode=argv[++i];else if(s=="--repeats")repeats=std::stoull(argv[++i]);else if(s=="--warmup")warmup=std::stoull(argv[++i]);else throw std::runtime_error("bad argument "+s);}bool branch=mode=="branched";if(path.empty()||(mode!="branched"&&mode!="control"))throw std::runtime_error("--input and valid --mode required");auto x=load(path);replay(x,warmup,branch);PMU pmu;pmu.start();auto t0=std::chrono::steady_clock::now();float checksum=replay(x,repeats,branch);auto t1=std::chrono::steady_clock::now();auto c=pmu.stop();uint64_t windows=x.size()/4,target=branch?windows*repeats*4:0;double rate=c.branches?double(c.misses)/c.branches:0,sec=std::chrono::duration<double>(t1-t0).count();std::cout<<"{\"mode\":\""<<mode<<"\",\"windows\":"<<windows<<",\"repeats\":"<<repeats<<",\"target_comparisons\":"<<target<<",\"branches\":"<<c.branches<<",\"branch_misses\":"<<c.misses<<",\"raw_branch_miss_rate\":"<<rate<<",\"seconds\":"<<sec<<",\"checksum\":"<<checksum<<"}\n";}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
