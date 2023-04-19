##Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.##

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

add_subdirectory(1)
add_subdirectory(3)


