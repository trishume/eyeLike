find_library(MGL mgl PATHS /usr/local/lib)

ADD_EXECUTABLE( eyeLike main.cpp )
TARGET_LINK_LIBRARIES( eyeLike ${OpenCV_LIBS} )
target_link_libraries( eyeLike ${MGL})