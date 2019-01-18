cmake_minimum_required(VERSION 3.5)

project(SHLight)
file (GLOB_RECURSE SOURCE *.cpp *.h *.hpp *.cc)

find_package (OpenCV REQUIRED)
find_package (ASSIMP REQUIRED)
find_package (EMBREE 2 REQUIRED)

set (EIGEN3_DIR D:/develop/library/eigen-3.3.4)

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${EIGEN3_DIR} 
    ${OpenCV_INCLUDE_DIRS} 
    ${ASSIMP_INCLUDE_DIRS} 
    ${EMBREE_INCLUDE_DIRS}
)

link_directories(
    ${ASSIMP_LIBRARY_DIRS}
)

add_executable(shlight ${SOURCE})

target_link_libraries(shlight  
    ${OpenCV_LIBRARIES}
    ${ASSIMP_LIBRARIES} 
    ${EMBREE_LIBRARIES}
)