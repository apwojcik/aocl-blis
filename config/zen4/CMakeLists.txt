##Copyright (C) 2022, Advanced Micro Devices, Inc ##

target_sources(${TARGET_NAME}
    PRIVATE
   		bli_cntx_init_zen4.c
		bli_family_zen4.h
   	)

target_include_directories(${TARGET_NAME}
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
	)