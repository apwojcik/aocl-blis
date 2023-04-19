##Copyright (C) 2020-2021, Advanced Micro Devices, Inc. All rights reserved.##

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_subdirectory(3)
add_subdirectory(1m)

