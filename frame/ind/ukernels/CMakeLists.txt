##Copyright (C) 2020, Advanced Micro Devices, Inc.## 

target_sources(${TARGET_NAME}
    PRIVATE
        bli_l3_ind_ukr.h
    )

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )