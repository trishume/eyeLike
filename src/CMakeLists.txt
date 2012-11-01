find_library(MGL mgl PATHS /usr/local/lib)

ADD_EXECUTABLE( eyeLike main.cpp findEyeCenter.cpp findEyeCenter.h constants.h)
TARGET_LINK_LIBRARIES( eyeLike ${OpenCV_LIBS} )
target_link_libraries( eyeLike ${MGL})