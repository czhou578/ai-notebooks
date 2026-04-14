This tells us convNeXt is running on a newer PyTorch (likely 2.8 or 2.9+). In these newer versions, the fused AdamW optimizer became stricter about enforcing that all internal tensors (params, grads, exp_avgs, exp_avg_sqs) share the same dtype.

When GradScaler is active, it internally calls scaler.unscale_(optimizer) before optimizer.step(), which manipulates gradient dtypes (FP16 → FP32). The older fused kernel tolerated this silently; the newer one rejects it with the explicit error you're seeing.

TL;DR: It "worked" in alexnet.ipynb only because that environment has an older PyTorch. It was always technically incorrect — the newer version just enforces the rule. Remove fused=True from convNeXt.ipynb and you should also remove it from alexnet.ipynb if you ever upgrade that environment.


---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
Cell In[4], line 130
    127     loss = loss_function(outputs, targets)
    129 scaler.scale(loss).backward()
--> 130 scaler.step(optimizer)
    131 scaler.update()
    133 optimizer.zero_grad(set_to_none=True)

File /usr/local/lib/python3.11/dist-packages/torch/amp/grad_scaler.py:440, in GradScaler.step(self, optimizer, *args, **kwargs)
    436     optimizer.grad_scale = (  # type: ignore[attr-defined]
    437         None if optimizer_state["stage"] == OptState.UNSCALED else scaler
    438     )
    439     optimizer.found_inf = found_inf  # type: ignore[attr-defined]
--> 440 retval = optimizer.step(*args, **kwargs_)
    441 optimizer_state["stage"] = OptState.STEPPED
    442 if not has_grad_scaler_kwarg:

File /usr/local/lib/python3.11/dist-packages/torch/optim/lr_scheduler.py:130, in LRScheduler.__init__.<locals>.patch_track_step_called.<locals>.wrap_step.<locals>.wrapper(*args, **kwargs)
    128 opt = opt_ref()
    129 opt._opt_called = True  # type: ignore[union-attr]
--> 130 return func.__get__(opt, opt.__class__)(*args, **kwargs)

File /usr/local/lib/python3.11/dist-packages/torch/optim/optimizer.py:484, in Optimizer.profile_hook_step.<locals>.wrapper(*args, **kwargs)
    479         else:
...
    697     torch._foreach_sub_(
    698         device_state_steps, [device_found_inf] * len(device_state_steps)
    699     )

RuntimeError: params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout



---------------------------------------------------------------------------

The Problem
torch.compile traces your model with "FakeTensors" to build an optimized graph. During tracing, it found that logits was on cuda:0 but targets was on cpu, causing:

Unhandled FakeTensor Device Propagation for aten.gather.default, found two different devices cuda:0, cpu

This happened because your original get_batch() never moved tensors to the GPU:

TorchRuntimeError                         Traceback (most recent call last)
Cell In[8], line 28
     25 m = torch.compile(m)
     26 optimizer = torch.optim.Adam(m.parameters(), lr=0.001, fused=True)
---> 28 logits, loss = m(xb, yb)
     29 scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda())

File /usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py:1553, in Module._wrapped_call_impl(self, *args, **kwargs)
   1551     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1552 else:
-> 1553     return self._call_impl(*args, **kwargs)

File /usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py:1562, in Module._call_impl(self, *args, **kwargs)
   1557 # If we don't have any hooks, we want to skip the rest of the logic in
   1558 # this function, and just call forward.
   1559 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1560         or _global_backward_pre_hooks or _global_backward_hooks
   1561         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1562     return forward_call(*args, **kwargs)
   1564 try:
   1565     result = None

File /usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py:433, in _TorchDynamoContext.__call__.<locals>._fn(*args, **kwargs)
    428 saved_dynamic_layer_stack_depth

----------------------------------------------------------------------------------------------




