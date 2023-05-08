#ifndef miniMOLS_TimerHelper_h
#define miniMOLS_TimerHelper_h

#include <fpu_control.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <iostream>

namespace PyLE {

    static double initRealTime = 0;
    static double initCpuTime = 0;
    static bool isInitialized = false;

    namespace TimerHelper {
      
        inline double cpuTime() {
          if(!isInitialized){
            std::cout << "Warning: initializeTime() has not been called before !" << std::endl;
          }
          struct rusage ru;
          getrusage(RUSAGE_SELF, &ru);
          if (initCpuTime != 0)
            return ((double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000) - initCpuTime;

          return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000; 
        }

        inline double realTime() {
          if(!isInitialized){
            std::cout << "Warning: initializeTime() has not been called before !" << std::endl;
          }
          struct timeval tv;
          gettimeofday(&tv, NULL);
          if (initRealTime != 0)
            return ((double)tv.tv_sec + (double) tv.tv_usec / 1000000) - initRealTime;
          
          return (double)tv.tv_sec + (double) tv.tv_usec / 1000000; 
        }

        inline void initializeTime(){
          initRealTime = 0;
          initCpuTime = 0;
          isInitialized = true;
          initRealTime = realTime();
          initCpuTime = cpuTime();
        }

        

        
    }
}
#endif