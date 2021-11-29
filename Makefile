all: programs

FWK_DIR        = libfwk/
FWK_MODE      ?= devel
FWK_GEOM      := disabled

CFLAGS        := -Isrc/
CFLAGS_linux  := -fopenmp
CFLAGS_mingw  := -fopenmp
LDFLAGS_linux := -fopenmp -Wl,--export-dynamic
LDFLAGS_mingw := -fopenmp

PCH_SOURCE    := src/lucid_pch.h
BUILD_DIR      = build/$(PLATFORM)_$(MODE)

include $(FWK_DIR)Makefile-shared

# --- Creating necessary sub-directories ----------------------------------------------------------

SUBDIRS        = build
BUILD_SUBDIRS  = 

ifndef JUNK_GATHERING
_dummy := $(shell mkdir -p $(SUBDIRS))
_dummy := $(shell mkdir -p $(addprefix $(BUILD_DIR)/,$(BUILD_SUBDIRS)))
endif

# --- List of source files ------------------------------------------------------------------------

SRC         := lucid_app lucid_renderer simple_renderer scene scene_setup scene_convert \
			   program shading texture_atlas wavefront_obj meshlet quad_generator
PROGRAM_SRC := lucid
ALL_SRC     := $(PROGRAM_SRC) $(SRC)

OBJECTS        := $(ALL_SRC:%=$(BUILD_DIR)/%.o)
PROGRAMS       := $(PROGRAM_SRC:%=%$(PROGRAM_SUFFIX))
programs:         $(PROGRAMS)

# --- Build targets -------------------------------------------------------------------------------

#time -o stats.txt -a -f "%U $@" 
$(OBJECTS): $(BUILD_DIR)/%.o:  src/%.cpp $(PCH_TARGET)
	$(COMPILER) -MMD $(CFLAGS_$*) $(CFLAGS) $(PCH_CFLAGS) -c $< -o $@

$(PROGRAMS): %$(PROGRAM_SUFFIX): $(SHARED_OBJECTS) $(BUILD_DIR)/%.o $(OBJECTS) $(FWK_LIB_FILE)
	$(LINKER) -o $@ $^ $(LDFLAGS) $(LDFLAGS_$*)

DEPS:=$(ALL_SRC:%=$(BUILD_DIR)/%.d) $(PCH_TEMP).d

JUNK_FILES    := $(OBJECTS) $(OBJ_molecular_cuda) $(DEPS)
JUNK_DIRS     := $(SUBDIRS)

# --- Other stuff ---------------------------------------------------------------------------------

# Recreates dependency files, in case they got outdated
depends: $(PCH_TARGET)
	@echo $(ALL_SRC) | tr '\n' ' ' | xargs -P16 -t -d' ' -I '{}' $(COMPILER) $(CFLAGS) $(PCH_CFLAGS) \
		src/'{}'.cpp -MM -MF $(BUILD_DIR)/'{}'.d -MT $(BUILD_DIR)/'{}'.o -E > /dev/null

-include $(DEPS)

