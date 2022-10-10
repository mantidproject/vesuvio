### Configure required dependencies if necessary
# Find git for everything
if(WIN32)
  set(_git_requires 1.9.5)
endif()
find_package(Git ${_git_requires})

set(MSVC_PYTHON_EXECUTABLE_DIR $ENV{CONDA_PREFIX})