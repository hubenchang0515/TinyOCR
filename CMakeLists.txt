cmake_minimum_required(VERSION 3.13)
project(tinyocr)

option(OCR_USE_GPU "use GPU" OFF)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(LIBS ${LIBS} ${OpenCV_LIBS})
endif()


if (OCR_USE_GPU)
    find_package(Vulkan)
    if (Vulkan_FOUND)
        include_directories(${Vulkan_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${Vulkan_LIBRARIES})
    else()
        message(WARNING "cannot find vulkan, guess: -lvulkan")
        set(LIBS ${LIBS} vulkan)
    endif()


    set(ENV{PKG_CONFIG_PATH} $prefix)
    find_package(PkgConfig)
    
    pkg_search_module(SPIRV spirv)
    if (SPIRV_FOUND)
        include_directories(${SPIRV_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${SPIRV_LIBRARIES})
    else()
        message(WARNING "cannot find spirv, guess: -lSPIRV -lSPIRV-Tools-opt -lSPIRV-Tools -lSPIRV-Tools-link -lglslang -lMachineIndependent -lOSDependent -lHLSL -lOGLCompiler -lGenericCodeGen -lSPVRemapper -lpthread")
        set(LIBS ${LIBS} SPIRV SPIRV-Tools-opt SPIRV-Tools SPIRV-Tools-link glslang MachineIndependent OSDependent HLSL OGLCompiler GenericCodeGen SPVRemapper pthread)
    endif()

    pkg_search_module(SPIRV_TOOLS SPIRV-Tools)
    if (SPIRV_FOUND)
        include_directories(${SPIRV_TOOLS_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${SPIRV_TOOLS_LIBRARIES})
    else()
        message(WARNING "cannot find SPIRV-Tools, guess: -lSPIRV-Tools-opt -lSPIRV-Tools -lSPIRV-Tools-link")
        set(LIBS ${LIBS} SPIRV-Tools-opt SPIRV-Tools SPIRV-Tools-link)
    endif()

    pkg_search_module(GLSL glslang)
    if (GLSL_FOUND)
        include_directories(${GLSL_INCLUDE_DIRS})
        set(LIBS ${LIBS} ${GLSL_LIBRARIES})
    else()
        message(WARNING "cannot find glslang, guess: -lglslang -lMachineIndependent -lOSDependent -lHLSL -lOGLCompiler -lGenericCodeGen -lSPVRemapper -lpthread")
        set(LIBS ${LIBS} glslang MachineIndependent OSDependent HLSL OGLCompiler GenericCodeGen SPVRemapper pthread)
    endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/usr/include)
link_directories(${CMAKE_SOURCE_DIR}/usr/lib)
set(LIBS ncnn ${LIBS})

include_directories(${CMAKE_SOURCE_DIR}/src)
aux_source_directory(${CMAKE_SOURCE_DIR}/src SRC)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/private PRIVATE_SRC)
aux_source_directory(${CMAKE_SOURCE_DIR}/src/demo DEMO_SRC)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -fopenmp)
add_library(${PROJECT_NAME} ${SRC} ${PRIVATE_SRC})
target_link_libraries(${PROJECT_NAME} ${LIBS})

add_executable(${PROJECT_NAME}-demo ${DEMO_SRC})
target_link_libraries(${PROJECT_NAME}-demo ${LIBS} ${PROJECT_NAME})

set(PUBLIC_HEADER
    ${CMAKE_SOURCE_DIR}/src/tinyocr.h
    ${CMAKE_SOURCE_DIR}/src/textarea.h
)

install(TARGETS 
        ${PROJECT_NAME} 
        ${PROJECT_NAME}-demo
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/TinyOCR
        )
