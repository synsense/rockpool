# def test_ffexpsyn():
#     # - Test FFExpSyn
# 
#     # from rockpool.layers import FFExpSynTorch
#     from rockpool.nn.layers import FFExpSyn
#     from rockpool.timeseries import TSEvent, TSContinuous
#     import numpy as np
# 
#     # - Layers
# 
#     size_in = 512
#     size = 3
#     dt = 0.001
# 
#     weights = np.linspace(-1, 1, size_in * size).reshape(size_in, size)
#     bias = np.linspace(-1, 1, size)
#     tau_syn = 0.15
#     # flT = FFExpSynTorch(weights, dt=dt, bias=bias, tau_syn=tau_syn)
#     flM = FFExpSyn(weights, dt=dt, bias=bias, tau_syn=tau_syn)
# 
#     # - Input signal
# 
#     tDur = 0.01
#     nSpikes = 5
# 
#     vnC = np.tile(np.arange(size_in), int(np.ceil(1.0 / nSpikes * size)))[:nSpikes]
#     vtT = np.linspace(0, tDur, nSpikes, endpoint=False)
#     tsIn = TSEvent(vtT, vnC, num_channels=size_in, t_stop=tDur)
# 
#     # - Evolve
#     # tsT = flT.evolve(tsIn)
#     tsM = flM.evolve(tsIn)
#     # flT.reset_all()
#     flM.reset_all()
# 
#     # assert(
#     #         np.isclose(tsT.samples, tsM.samples, rtol=1e-4, atol=1e-5).all()
#     #     # and np.isclose(tsT.times, tsM.times).all()
#     # ), "Layer outputs are not the same."
# 
#     # - Training (only FFExpSyn and FFExpSynTorch)
#     mfTgt = np.array(
#         [
#             np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
#             for fPhase in np.linspace(0, 3, size)
#         ]
#     ).T
#     tsTgt = TSContinuous(np.arange(int(tDur / dt)) * dt, mfTgt)
# 
#     # flT.train_rr(tsTgt, tsIn, regularize=0.1, is_first=True, is_last=True)
#     flM.train_rr(
#         tsTgt,
#         tsIn,
#         regularize=0.1,
#         is_first=True,
#         is_last=True,
#         return_trained_output=True,
#     )
#     # Test training without biases
#     flM.train_rr(
#         tsTgt,
#         tsIn,
#         regularize=0.1,
#         is_first=True,
#         is_last=True,
#         return_trained_output=True,
#         train_biases=False,
#     )
# 
#     # assert(
#     #             np.isclose(flT.weights, flM.weights, rtol=1e-4, atol=1e-2).all()
#     #         and np.isclose(flT.bias, flM.bias, rtol=1e-4, atol=1e-2).all()
#     # ), "Training led to different results"
# 
# 
# # def test_ffexpsyntorch():
# #     # - Test FFIAFTorch
# #
# #     # from rockpool.layers import FFExpSynTorch
# #     from nn.layers import FFExpSyn
# #     from nn.layers import FFExpSynTorch
# #     from rockpool.timeseries import TSEvent, TSContinuous
# #     import numpy as np
# #
# #     # - Layers
# #
# #     size_in = 512
# #     size = 3
# #     dt = 0.001
# #
# #     weights = np.linspace(-1, 1, size_in * size).reshape(size_in, size)
# #     bias = np.linspace(-1, 1, size)
# #     tau_syn = 0.15
# #     flT = FFExpSynTorch(weights, dt=dt, bias=bias, tau_syn=tau_syn)
# #     flM = FFExpSyn(weights, dt=dt, bias=bias, tau_syn=tau_syn)
# #
# #     # - Input signal
# #
# #     tDur = 0.01
# #     nSpikes = 5
# #
# #     vnC = np.tile(np.arange(size_in), int(np.ceil(1.0 / nSpikes * size)))[:nSpikes]
# #     vtT = np.linspace(0, tDur, nSpikes, endpoint=False)
# #     tsIn = TSEvent(vtT, vnC, num_channels=size_in, t_stop=tDur)
# #
# #     # - Evolve
# #     try:
# #         tsT = flT.evolve(tsIn)
# #     # - Catch runtime error ("code is too big") that occurs on the gitlab server
# #     except RuntimeError:
# #         return
# #     else:
# #         tsM = flM.evolve(tsIn)
# #         flT.reset_all()
# #         flM.reset_all()
# #
# #         assert (
# #             np.isclose(tsT.samples, tsM.samples, rtol=1e-4, atol=1e-5).all()
# #             # and np.isclose(tsT.times, tsM.times).all()
# #         ), "Layer outputs are not the same."
# #
# #         # - Training
# #         mfTgt = np.array(
# #             [
# #                 np.sin(np.linspace(0, 10 * tDur, int(tDur / dt)) + fPhase)
# #                 for fPhase in np.linspace(0, 3, size)
# #             ]
# #         ).T
# #         tsTgt = TSContinuous(np.arange(int(tDur / dt)) * dt, mfTgt)
# #
# #         flT.train_rr(tsTgt, tsIn, regularize=0.1, is_first=True, is_last=True)
# #         flM.train_rr(tsTgt, tsIn, regularize=0.1, is_first=True, is_last=True)
# #
# #         assert (
# #             np.isclose(flT.weights, flM.weights, rtol=1e-4, atol=1e-2).all()
# #             and np.isclose(flT.bias, flM.bias, rtol=1e-4, atol=1e-2).all()
# #         ), "Training led to different results"
# #
# 
# # def test_rr_single_output():
# #     from training.train_rr import RidgeRegrTrainer
# #     import numpy as np
# #
# #     model = lambda X, beta, sig: np.matmul(X, beta) + np.random.normal(
# #         loc=0, scale=sig, size=(len(X), 1)
# #     )
# #
# #     num_samples = 200
# #     num_coef = 50
# #     sig = 0.1
# #
# #     X = np.random.rand(num_samples, num_coef)
# #     beta = np.random.rand(num_coef, 1)
# #
# #     y = model(X, beta, sig)
# #
# #     rr = RidgeRegrTrainer(
# #         num_features=num_coef,
# #         num_outputs=1,
# #         regularize=0.1,
# #         fisher_relabelling=False,
# #         standardize=False,
# #         train_biases=False,
# #     )
# #
# #     num_batches = 2
# #     num_samples_per_batch = num_samples // num_batches
# #     for i in range(num_batches):
# #         batch_start = i * num_samples_per_batch
# #         batch_end = (i + 1) * num_samples_per_batch
# #         rr.train_batch(X[batch_start:batch_end, :], y[batch_start:batch_end])
# #
# #     rr.update_model()
# #
# #
# # def test_rr_single_output_pruning():
# #     from training.train_rr import RidgeRegrTrainer
# #     import numpy as np
# #
# #     model = lambda X, beta, sig: np.matmul(X, beta) + np.random.normal(
# #         loc=0, scale=sig, size=(len(X), 1)
# #     )
# #
# #     num_samples = 200
# #     num_coef = 50
# #     sig = 0.1
# #
# #     n_prune = 10
# #
# #     X = np.random.rand(num_samples, num_coef)
# #     beta = np.random.rand(num_coef, 1)
# #
# #     y = model(X, beta, sig)
# #
# #     rr = RidgeRegrTrainer(
# #         num_features=num_coef,
# #         num_outputs=1,
# #         regularize=0.1,
# #         fisher_relabelling=False,
# #         standardize=False,
# #         train_biases=False,
# #     )
# #
# #     rr.train_batch(X, y)
# #     rr.update_model(n_prune=n_prune)
# #
# #     rr.train_batch(X, y)
# #     rr.update_model()
# #
# #     assert np.where(rr.weights == 0)[0].size >= n_prune
# #     y_pred = model(X, rr.weights, 0)
# #
# #     mse = np.mean((y - y_pred) ** 2)
# #
# #
# # def test_rr_multi_output_batched_pruning():
# #     from training.train_rr import RidgeRegrTrainer
# #     import numpy as np
# #
# #     model = lambda X, beta, sig: np.matmul(X, beta) + np.random.normal(
# #         loc=0, scale=sig, size=(len(X), 1)
# #     )
# #
# #     num_epochs = 3
# #     num_samples = 200
# #     num_coef = 50
# #     num_out = 2
# #     sig = 0.1
# #
# #     n_prune = 10
# #
# #     X = np.random.rand(num_samples, num_coef)
# #     beta = np.random.rand(num_coef, num_out)
# #
# #     y = model(X, beta, sig)
# #
# #     rr = RidgeRegrTrainer(
# #         num_features=num_coef,
# #         num_outputs=num_out,
# #         regularize=0.1,
# #         fisher_relabelling=False,
# #         standardize=False,
# #         train_biases=False,
# #     )
# #
# #     rr.train_batch(X, y)
# #     for i in range(num_epochs):
# #         rr.update_model(n_prune=n_prune * (i + 1))
# #         rr.train_batch(X, y)
# #     rr.update_model()
# #
# #     pruned_ids = np.where(rr.weights == 0)[0]
# #
# #     # assert that number of pruned neurons is at least n_prune * num_epochs * num_out
# #     assert pruned_ids.size >= n_prune * num_epochs * num_out
# #
# #
# # def test_rr_pruning_quality():
# #     from training.train_rr import RidgeRegrTrainer
# #     import numpy as np
# #     import copy
# #
# #     model = lambda X, beta, sig: np.matmul(X, beta) + np.random.normal(
# #         loc=0, scale=sig, size=(len(X), 1)
# #     )
# #
# #     num_samples = 100000
# #     num_coef = 50
# #     num_out = 1
# #     sig = 0.1
# #     regularize = 0.1
# #
# #     n_prune = 10
# #
# #     for iteration in range(10):
# #         # repeat this 100 times
# #         X = np.random.rand(num_samples, num_coef)
# #         beta = np.random.rand(num_coef, num_out)
# #
# #         # try to trick the approach
# #         beta[: int(n_prune / 2)] /= 100
# #
# #         y = model(X, beta, sig)
# #
# #         rr = RidgeRegrTrainer(
# #             num_features=num_coef,
# #             num_outputs=num_out,
# #             regularize=regularize,
# #             fisher_relabelling=False,
# #             standardize=False,
# #             train_biases=False,
# #         )
# #
# #         rr.train_batch(X, y)
# #         rr.update_model(n_prune=n_prune)
# #         rr.train_batch(X, y)
# #         rr.update_model()
# #
# #         y_pred = model(X, rr.weights, 0)
# #         mse = np.mean((y - y_pred) ** 2)
# #
# #         rr_naive = RidgeRegrTrainer(
# #             num_features=num_coef,
# #             num_outputs=num_out,
# #             regularize=regularize,
# #             fisher_relabelling=False,
# #             standardize=False,
# #             train_biases=False,
# #         )
# #
# #         rr_naive.train_batch(X, y)
# #         rr_naive.update_model()
# #         # prune n_prune weakest weights and remove from them also from X
# #         ids_to_prune = np.argsort(np.ravel(rr_naive.weights))[:n_prune]
# #         X_retrain = copy.deepcopy(X)
# #         X_retrain[:, ids_to_prune] = 0
# #         rr_naive.train_batch(X_retrain, y)
# #         rr_naive.update_model()
# #
# #         rr_naive.weights[ids_to_prune, 0] = 0
# #
# #         y_pred_naive = model(X, rr_naive.weights, 0)
# #         mse_naive = np.mean((y - y_pred_naive) ** 2)
# #
# #         rr_naive_rand = RidgeRegrTrainer(
# #             num_features=num_coef,
# #             num_outputs=num_out,
# #             regularize=regularize,
# #             fisher_relabelling=False,
# #             standardize=False,
# #             train_biases=False,
# #         )
# #
# #         rr_naive_rand.train_batch(X, y)
# #         rr_naive_rand.update_model()
# #         # prune n_prune random weights and remove from them also from X
# #         ids_to_prune = np.random.choice(range(rr_naive_rand.weights.size), n_prune)
# #         X_retrain = copy.deepcopy(X)
# #         X_retrain[:, ids_to_prune] = 0
# #         rr_naive_rand.train_batch(X_retrain, y)
# #         rr_naive_rand.update_model()
# #
# #         rr_naive_rand.weights[ids_to_prune, 0] = 0
# #
# #         y_pred_naive_rand = model(X, rr_naive_rand.weights, 0)
# #         mse_naive_rand = np.mean((y - y_pred_naive_rand) ** 2)
# #
# #         assert mse <= mse_naive
# #         assert mse <= mse_naive_rand
# #
# #
# # def test_rr_bias_pruning():
# #     """ biases should be not pruned! """
# #     from training.train_rr import RidgeRegrTrainer
# #     import numpy as np
# #
# #     model = lambda X, beta, sig: np.matmul(X, beta) + np.random.normal(
# #         loc=0, scale=sig, size=(len(X), 1)
# #     )
# #
# #     num_epochs = 3
# #     num_samples = 200
# #     num_coef = 50
# #     num_out = 2
# #     sig = 0.1
# #
# #     n_prune = 49
# #
# #     X = np.random.rand(num_samples, num_coef)
# #     beta = np.random.rand(num_coef, num_out)
# #
# #     y = model(X, beta, sig)
# #
# #     rr = RidgeRegrTrainer(
# #         num_features=num_coef,
# #         num_outputs=num_out,
# #         regularize=0.1,
# #         fisher_relabelling=False,
# #         standardize=False,
# #         train_biases=True,
# #     )
# #
# #     rr.train_batch(X, y)
# #     rr.update_model(n_prune=n_prune)
# #
# #     # assert that no biases got pruned
# #     assert np.all(rr.bias != 0)
# #
# #
# # def test_rr_output_wise_pruning():
# #
# #     from training.train_rr import RidgeRegrTrainer
# #     import numpy as np
# #
# #     model = lambda X, beta, sig: np.matmul(X, beta) + np.random.normal(
# #         loc=0, scale=sig, size=(len(X), 1)
# #     )
# #
# #     num_epochs = 3
# #     num_samples = 200
# #     num_coef = 50
# #     num_out = 2
# #     sig = 0.1
# #
# #     n_prune = 10
# #
# #     X = np.random.rand(num_samples, num_coef)
# #     beta = np.random.rand(num_coef, num_out)
# #
# #     y = model(X, beta, sig)
# #
# #     rr = RidgeRegrTrainer(
# #         num_features=num_coef,
# #         num_outputs=num_out,
# #         regularize=0.1,
# #         fisher_relabelling=False,
# #         standardize=False,
# #         train_biases=False,
# #     )
# #
# #     rr.train_batch(X, y)
# #     rr.update_model(n_prune=n_prune)
# #
# #     pruned_ids = np.where(rr.weights == 0)
# #
# #     # assert that each output channel pruned n_prune weights
# #     assert len(np.where(pruned_ids[1] == 0)[0]) == n_prune
# #     assert len(np.where(pruned_ids[1] == 1)[0]) == n_prune
# #
# #
# # def test_rr_correct_pruning():
# #     from training.train_rr import RidgeRegrTrainer
# #     import numpy as np
# #
# #     model = lambda X, beta, sig: np.matmul(X, beta) + np.random.normal(
# #         loc=0, scale=sig, size=(len(X), 1)
# #     )
# #
# #     num_epochs = 3
# #     num_samples = 200
# #     num_coef = 50
# #     num_out = 2
# #     sig = 0.1
# #
# #     n_prune = 10
# #
# #     X = np.random.rand(num_samples, num_coef) + 0.1
# #     beta = np.random.rand(num_coef, num_out)
# #
# #     # this should prune the first dimensions
# #     X[:, :n_prune] /= 100
# #     X[:, n_prune:] *= 100
# #
# #     y = model(X, beta, sig)
# #
# #     rr = RidgeRegrTrainer(
# #         num_features=num_coef,
# #         num_outputs=num_out,
# #         regularize=0.1,
# #         fisher_relabelling=False,
# #         standardize=False,
# #         train_biases=False,
# #     )
# #
# #     rr.train_batch(X, y)
# #     rr.update_model(n_prune=n_prune)
# #     rr.train_batch(X, y)
# #     rr.update_model()
# #
# #     pruned_ids = np.where(rr.weights == 0)
# #
# #     # assert that each pruned ids are the first n_prune inputs
# #     assert np.all([p in range(n_prune) for p in pruned_ids[0]])
# 
