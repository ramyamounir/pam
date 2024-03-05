import sys; sys.path.append('./')
from experiments.sequence_memory.solo import run_experiments as run_solo
from experiments.sequence_memory.lstm import run_experiments as run_lstm
import dask
from dask import delayed


# run_solo(K_range=[8,12,16], test_types=['AP', 'AG', 'NA'], epochs=200)
# run_solo(K_range=[16], test_types=['AP'], epochs=500)
# run_lstm(L_range=[1], test_types=['AP'], epochs=200)

delayed_lstm = delayed(run_lstm)
delayed_solo = delayed(run_solo)

result1 = delayed_lstm(L_range=[1], test_types=['AP'], epochs=200)
result2 = delayed_solo(K_range=[8,12,16], test_types=['AP', 'AG', 'NA'], epochs=200)

results = dask.compute(result1, result2)

