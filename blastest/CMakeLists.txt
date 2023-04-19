##Copyright (C) 2020, Advanced Micro Devices, Inc. All rights reserved.##

set(F2C_LIB "libf2c")

include_directories(f2c)

# Generate F2C library
add_library(${F2C_LIB} STATIC )
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE C)

add_subdirectory(f2c)
add_subdirectory(src)
