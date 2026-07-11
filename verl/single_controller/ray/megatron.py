
from typing import Optional

import ray

from verl.single_controller.base.megatron.worker import DistGlobalInfo, DistRankInfo
from verl.single_controller.base.megatron.worker_group import MegatronWorkerGroup

from .base import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup


class NVMegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    def __init__(self, resource_pool: RayResourcePool, ray_cls_with_init: RayClassWithInitArgs, **kwargs):
        """
        Initialize the NVMegatronRayWorkerGroup.

        Args:
            resource_pool (RayResourcePool): The resource pool containing worker resources
            ray_cls_with_init (RayClassWithInitArgs): The Ray class with initialization arguments
            **kwargs: Additional keyword arguments to pass to the parent class
        """
        super().__init__(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, **kwargs)
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name="get_megatron_rank_info")
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name="get_megatron_global_info")
        )


class MegatronRayWorkerGroup(RayWorkerGroup, MegatronWorkerGroup):
    """
    MegatronWorkerGroup will query each worker of its megatron rank info and store it inside the WorkerGroup
    so that the dispatcher can use it to dispatch data.
    """

    def __init__(
        self,
        resource_pool: RayResourcePool,
        ray_cls_with_init: RayClassWithInitArgs,
        default_megatron_kwargs: dict = None,
        **kwargs,
    ):
        super().__init__(
            resource_pool=resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            default_megatron_kwargs=default_megatron_kwargs,
            **kwargs,
        )
        self.init_megatron(default_megatron_kwargs=default_megatron_kwargs)
        self._megatron_rank_info: DistRankInfo = self.execute_all_sync(method_name="get_megatron_rank_info")
        self._megatron_global_info: DistGlobalInfo = ray.get(
            self.execute_rank_zero_async(method_name="get_megatron_global_info")
        )

    def init_megatron(self, default_megatron_kwargs: Optional[dict] = None):
        if not self._is_init_with_detached_workers:
            self.execute_all_sync(method_name="init_megatron", default_megatron_kwargs=default_megatron_kwargs)
