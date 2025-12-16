cmake_minimum_required(VERSION 4.0)
project(HPCA CUDA)

set(CMAKE_CXX_STANDARD 14)

add_executable(HPCA
        main.cu)

target_compile_options(
        HPCA
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)