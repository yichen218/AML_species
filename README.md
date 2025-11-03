pip install -r requirements.txt
python scripts/build\_features.py

grid\_group only for evaluation, not for training models



global\_distribution.png — 全球散点（观测点分布）
spatial\_hotspots\_density.png — Hexbin 密度热点
lat\_distribution.png — 纬度直方图
lon\_distribution.png — 经度直方图
lat\_band\_distribution.png — 纬度带计数
hemisphere\_distribution.png — 南/北半球分布



taxon\_id (int, 仅 train)：物种标签（很多任务的 y 或输入 ID）

lat (float)、lon (float)：原始坐标（用于地点→物种、评估或可视化）

grid\_lat (int)、grid\_lon (int)：1° 栅格索引（地点→物种可作为数值特征）

grid\_group (str)："{grid\_lat}\_{grid\_lon}"（仅用于分组/切分，不可作特征）

lat\_band (category)：六段纬度带（目标或派生 OHE 的来源）

hemisphere (category)：N/S（目标或派生 OHE 的来源）

sin\_lon (float)、cos\_lon (float)：经度周期编码（地点→物种常用）

taxon\_name (str/空)（展示/映射用，不训练）

