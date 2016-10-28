cmake_minimum_required(VERSION 2.8)

project(samplePairwise)

add_executable(pairwise main.cpp)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm.*|ARM.*)")
    set(ARM 1)
endif()

if (ARM)
    if (UNIX)
        set(CMAKE_CXX_FLAGS "-mfpu=neon-fp16")
    endif()
endif()

