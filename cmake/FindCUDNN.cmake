find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h REQUIRED)
find_library(CUDNN_LIBRARY NAMES cudnn REQUIRED)

if(NOT TARGET CUDNN::cudnn_all)
    add_library(CUDNN::cudnn_all UNKNOWN IMPORTED)
    set_target_properties(CUDNN::cudnn_all PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
    )
endif()

# MLX uses cuDNN Frontend as a header-only CMake target. Its installed static
# package references that target but does not export it, which otherwise turns
# into an invalid -lcudnn_frontend linker flag in consuming projects.
if(NOT TARGET cudnn_frontend)
    add_library(cudnn_frontend INTERFACE)
endif()

set(CUDNN_FOUND TRUE)
