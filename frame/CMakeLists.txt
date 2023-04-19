##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

add_subdirectory(thread)
add_subdirectory(base)
add_subdirectory(0)
add_subdirectory(1)
add_subdirectory(1d)
add_subdirectory(1f)
add_subdirectory(1m)
add_subdirectory(2)
add_subdirectory(3)
add_subdirectory(compat)
add_subdirectory(ind)
add_subdirectory(util)

target_include_directories(${TARGET_NAME}
    PRIVATE
        include
        include/level0
        include/level0/1e
        include/level0/1m
        include/level0/1r
        include/level0/bb
        include/level0/io
        include/level0/ri
        include/level0/ri3
        include/level0/rih
        include/level0/ro
        include/level0/rpi
    )




