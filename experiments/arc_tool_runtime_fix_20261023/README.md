# 工具运行修复：本地待评审版本，2026-10-23

已完成代码修改及本地 CPU 合成测试：51 项 Python 测试、9 项 Node 控制器测试通过。
没有启动 GPU、运行已保存的 ARC 程序、修改比赛提交或更新正在训练/评测的云端代码包。

## 已改内容

1. `src/arc_feedback_harness/worker.py` 的 stdout/stderr 累计保存最多 4,000 字符；超出后设置 `stdout_truncated=true` 并丢弃多余文本，继续执行脚本，不再抛异常。缓冲区填满后不再追加空字符串。
2. 允许生成器、普通类、global/nonlocal、装饰器和 `dtype=object`。不支持的公开 import 在运行时抛标准 `ImportError`，可以进入模型写出的 `except ImportError` 分支。补齐 `ImportError` 等异常类型。
3. 可用且兼容的 `scipy.ndimage` 在子进程隔离前加载；未安装时仍可使用手写 BFS。`sys` 只提供递归深度设置/查询（最大10,000），不暴露 stdin/stdout/modules/path。文件、网络、进程与反射限制及 Linux seccomp/rlimits 均保留。
4. 新的 `harness/agent.mjs` / `panel.mjs` 使用可配置 `pythonLimit` / `--python-limit`，此本地版本默认24。保持 Medium、累计65K、最大并发24、3次正式提交；没有单轮16K截断。提示词仅按实际额度替换原来的12，其他措辞不变。
5. 保留现有经过修正的超额 Python 路由和 verified snapshot 收尾；失败超额调用不扣 `final_checks`。保持首个 eligible 即结束，没有第二次冷启动搜索。

当前 A/B 的冻结 ZIP `1cyQTprkzcX_2yJ7MUldU18igqSA4LJv_` 未变。
新包是下一版代码准备，不会被 Stage2 v3 当前作业自动使用。该作业仍用12次Python，以保持与已完成A组的对照一致。

## 从保存日志得到的事实

`audit_saved.py` 只读取已保存 JSON/JSONL 和评分结果，未执行任何候选。
完整逐条证据在 `analysis/SAVED_AUDIT.json`。

- Stage1 65K 的375条桥接执行记录中，Python普通执行337次，额外路由探测9次（其中1次合格），正式检查计数30（包含1次合格路由）。9次路由中8次失败均未扣正式额度。
- 该轮 `tool_errors=45`：10次stdout中断、1次import限制、8次超额阻断、26次普通Python错误。可直接处理的前三类为19/45（42.2%），不是“超过一半”。
- 原90K的 summary 记录50次执行错误，而 Pi事件含69次 `isError`，还包括17次额度提醒与其他控制器阻断。历史表里的50与69是不同统计口径。
- Stage2 v2 的 summary为70、Pi事件74；旧错误路由里有14次未通过的超额检查被计入正式额度。该问题在A组已经消除。
- 原90K复现与历史90K的48个首轮中，8个思考/正文文本完全相同；加上工具名和代码参数后仍有7个完全相同（忽略生成的工具ID、时间戳、usage）。因此“0/48首轮相同”不符合保存内容。文本前缀长度中位数425.5字符；本审计没有读取logits，不能报 `1e-6` 实测误差。
- 五组整题oracle并集29，五组都解出的交集6，四组以上解出11。三组16分的两两并集依次为20（复现+Stage1）、23（复现+Stage2 v2）、20（Stage1+Stage2 v2）。这些组的模型/预算/harness并不全部相同；29是已观测互补范围，不是两个槽位的稳定得分。
- 在Stage1路由修复和Stage2 v2两组含快照的保存记录中，没有同题两套不同且demo全过的已保存预测。单条轨迹候选复用有价值，但现有记录没有可直接加入第二槽位的额外合格答案；不为此启动第二条推理链。

关于相同seed仍有运行波动，vLLM官方说明可见：
https://docs.vllm.ai/en/latest/usage/reproducibility/
目前实际同配置复现结果是22→16；真实均值16–18、逐题35%–50%成功率都还不是现有同协议重复运行的测量结果。

## 按240题计算，而不是120题

以启动/保存之外有40,300秒（约11.2h）生成时间为规划假设：

- 248 token/s允许总输出9,994,400 token，平均每题41,643 token。
- 本次Stage1 65K实际输出2,412,291/48 = 50,256 token/题。
- 按此平均量做240题需要12,061,455 token；248 token/s光生成约13.51h，加启动/保存约14.32h。
- 对应所需面板吞吐约299.3 token/s；或者保持248 token/s，把实际平均输出降约17.1%。65K硬上限不等于每题实际65K，更不等于能自动在12小时完成。
- 提高工具次数增加的不只有CPU执行。每个反馈之后还会生成新token，并可能重算丢失的前缀；65K总上限保留，但实际平均消耗及吞吐仍要在下一次获准评测时测量。

已保存L4两组实测：单次80K配置2,552,517输出/10,461.074秒=244.00token/s；44K+36K配置2,832,848/11,264.38秒=251.49token/s。对应结果14/48、20/67与14/48、18/67。历史自动报告的“65K”表头是复用标题，不能据此覆盖实验配置身份。

## 测试与文件

- `tests/test_tool_runtime_fixes.py`：输出截断后继续验证/保存、缺包catch、ndimage导入形式、生成器/类/装饰器/闭包、受限sys、原有外部访问限制。
- `test_overflow.py`：当前调用无transform/提取失败/demo失败/query非法均保留正式额度，原提示完整返回。
- `test_snapshot.mjs`：24次额度与提示一致、全程共享65K、3次正式提交仍可用、超额失败不扣额度、快照与子进程异常收尾。
- Python测试使用仓库已有Python3.12.14环境，NumPy2.5.2、SciPy1.18.1；Node24.12.0。均为手写小网格/模拟流，不是模型生成或Linux生产隔离验证。
- `SOURCE_REVIEW.zip` 仅包含本地新harness、共享Python模块、配置与本说明，不含权重/轨迹/gold/部署启动Notebook。

## 现有云端作业

两台已完成评测的SAVE_VERIFIED/DEPLOYMENT_DONE已读回，本地现有ZIP复用校验bytes/members/CRC，Colab Manage sessions独立确认两台均消失。
Stage2 v3截至14:02UTC已25/64更新，仍用原冻结配置，继续按同一12小时CPU起点运行SFT→合并→评测；硬截止23:44:35.786800UTC（次日JST08:44:35）。
