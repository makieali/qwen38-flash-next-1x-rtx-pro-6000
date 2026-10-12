import time, torch, flashinfer
t = time.time()
p = torch.softmax(torch.randn(4, 248320, device="cuda"), -1)
print("sample:", flashinfer.sampling.top_k_top_p_sampling_from_probs(p, 20, 0.8).tolist(), "in %.0fs" % (time.time() - t), flush=True)
