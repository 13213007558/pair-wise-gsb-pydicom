# uid_remap — 批量 DICOM 实例 UID 替换器

只负责一件事：把测试数据里的**实例身份 UID**（Study / Series / SOP Instance /
Frame of Reference 及其各类引用）整批换成新身份，同时保持文件之间的引用闭合。
不做匿名化、不改病人信息、不动像素，除此之外的标签一律原样保留。

## 替换规则

- **替换**：uid_remap/uidtags.py 中 INSTANCE_UID_TAGS 列出的实例身份 UID
  （含 ReferencedSOPInstanceUID 等引用），序列（SQ）递归处理，多值 UI 逐值替换。
- **不动**：DEFINITION_UID_TAGS 中的定义类 UID —— SOP Class、Transfer Syntax、
  Coding Scheme、Implementation/Device 等。见到 UI 类型不会一律替换。
- **File Meta 同步**：MediaStorageSOPInstanceUID (0002,0003) 强制与正文
  SOPInstanceUID (0008,0018) 一致；源文件不一致时以正文为准并记 warning。
- **同一旧身份 → 同一新身份**：映射在整个批次内全局生效，出现多少次都一样。
- **未分类私有 UI**：不在两张表内的 UI 字段（通常是私有 tag）不修改，列入
  报告的 unclassified_private_ui，等调用方补充分类规则后再处理。

## 修改前校验（任一失败则整批不写盘）

- 分配函数不稳定：同一旧 UID 两次调用结果不同；
- 新 UID 非法：非数字/点、组件前导零、超过 64 字符；
- 新 UID 冲突：两个旧 UID 映射到同一个新 UID、新 UID 撞上批内另一个旧身份、
  或撞上批内出现的定义类 UID。

所有问题汇总为一个 RemapValidationError 一次性抛出。

## API

    from uid_remap import remap_batch, hash_assigner, load_mapping

    result = remap_batch(
        inputs,                 # 输入文件路径列表
        output_dir,             # 输出目录（不与输入混放）
        assign=hash_assigner(root="2.25", salt="batch-1"),  # 调用者给的旧->新分配函数
        mapping_in=load_mapping("prev/uid_mapping.json"),   # 可选：上一批的映射
    )
    result.mapping        # 合并后的 旧->新 映射，导出供下一批继续使用
    result.changes        # 改动清单（文件 / 标签路径 / 旧值 / 新值）
    result.unclassified   # 未分类私有 UI 字段（未修改）
    result.warnings       # 例如 File Meta 与正文不一致被强制同步

assign 是任意 callable(old_uid) -> new_uid，必须确定性强；
hash_assigner 是内置的确定性实现（2.25.<sha256 十进制>）。

## CLI

    python3 -m uid_remap IN_DIR OUT_DIR [--mapping-in prev.json]
        [--mapping-out uid_mapping.json] [--report remap_report.json]
        [--root 2.25] [--salt batch-1]

输出目录中包含替换后的文件、uid_mapping.json（供下一批 --mapping-in）
和 remap_report.json（改动清单 + 未分类私有 UI + 警告）。

## 验证

tests/make_synthetic.py 生成三份纯合成、互相引用的文件
（两张影像共享 Study/Series/FrameOfReference，一张结构集经三层嵌套序列
引用两者，另含多值 UI、CodingSchemeUID 和一个私有 UI tag）：

    python3 -m unittest tests.test_uid_remap -v

覆盖：引用闭合、File Meta 同步、定义类 UID 不变、私有 UI 只报不改、
映射跨批复用幂等、冲突/非法/不稳定映射在写盘前中止、隐式 VR 与
大端传输语法往返。

## 环境说明

引擎使用仓库自带的 pydicom 源码（src/）。离线环境没有安装 pydicom，
uid_remap/_pydicom.py 会把 src/ 和 .pymeta/（最小 dist-info 垫片，
仅为满足 importlib.metadata.version("pydicom")）加入 sys.path；
若运行环境已安装 pydicom 则直接使用安装版。

