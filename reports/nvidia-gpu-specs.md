# NVIDIA GPU Specs MVP

## 目标与范围
- 覆盖 6 个架构层记录: Volta, Turing, Ampere, Ada, Hopper, Blackwell
- 覆盖 17 个代表 SKU: Tesla V100, Tesla T4, TITAN RTX, Quadro RTX 6000, A100, A40, RTX A6000, GeForce RTX 3090, L4, L40S, RTX 6000 Ada, GeForce RTX 4090, H100, H200, B200, RTX PRO 6000 Blackwell Workstation Edition, GeForce RTX 5090
- 只以 NVIDIA 官方白皮书、产品页、datasheet/brief、CUDA compute capability 页面为主依据

## 官方来源优先级
1. CUDA compute capability 列表
2. 架构白皮书 / architecture pages
3. SKU 官方产品页 / datasheet / product brief
4. 第三方补充来源仅允许出现在 supplemental_sources，本次生成默认未使用

## 字段与 null 策略
- 所有字段都保留 source URL 和 evidence excerpt
- 无法从官方材料直接确认的字段写 null
- 只有官方明确给出“不支持/没有”时才应使用 0；本次 MVP 默认避免推断型 0

## 更新方式
```bash
conda run -n torch python scripts/manage-nvidia-specs.py refresh
```

## 当前覆盖概况
- 记录总数: 23
- 成功抓取官方来源: 27/27
- 有官方 tensor throughput 的记录: 11
- supplemental source 使用数: 0

## 已知缺口
- Hopper / Blackwell 的公开页面更强调 capability 与 throughput，公开拓扑和 SKU 启用单元数相对少
- 消费级 Blackwell SKU 常公开 CUDA core / AI TOPS / Tensor gen，但未稳定公开 SM/Tensor/RT 精确数量与 die family
- B200 当前通过 DGX 系统页最容易拿到官方信息，因此 per-GPU 单元数保守留空

## Missing-Field Preview
- B200: 19 missing fields
- Quadro RTX 6000: 17 missing fields
- A40: 16 missing fields
- GeForce RTX 3090: 16 missing fields
- GeForce RTX 4090: 16 missing fields
- Blackwell: 15 missing fields
- RTX 6000 Ada: 15 missing fields
- RTX A6000: 15 missing fields
