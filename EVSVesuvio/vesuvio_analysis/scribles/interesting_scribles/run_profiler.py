
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
import My_edited_backward_script
profiler.disable()

stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()




