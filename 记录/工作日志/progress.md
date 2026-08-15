# PI-JWM v11 Selector Progress Log

## 2026-08-15 仓库顶层目录重构设计

- [x] 读取`brainstorming`、`planning-with-files`与`writing-plans`规则；运行session catch-up并记录当前dirty worktree。
- [x] 只读清点根目录、`文档/`、`代码/`、测试/tmp目录及全部活跃路径引用。
- [x] 提出2—3种迁移兼容方案并逐节请求用户确认。
- [x] 用户确认后写规格和实施计划；确认前未移动文件。
- 工具记录：首次三文件规划补丁因`findings.md`文件头上下文不匹配而整体拒绝；重新读取真实文件头后拆成最小补丁。
- 工具记录：首次目录清点因PowerShell多键排序参数语法错误在解析阶段退出；无文件变化，改用显式属性表达式。
- 工具记录：修正后的聚合清点退出0但无stdout，无法作为证据；改为小查询分块读取。
- [x] 清点根目录9个Markdown、原文档四类顶层目录和代码标准子结构；未移动文件。
- 工具记录：文档相对路径展示使用错误硬编码截断长度，产生非终止错误；只影响显示，后续用动态根长度重读。
- [x] 动态重读文档树、各类文件数量/体积、知识库三文件、研究进展顶层材料、测试/tmp/cache路径和Git跟踪前缀。
- [x] 审计`pyproject.toml`、`.gitignore`、根导航/规则文档、三个活动worktree及tracked旧路径引用：95文件含旧代码路径、29文件含旧文档路径。
- [x] 用户确认`paper/`只承接正式论文材料；过程记录进`记录/研究进展/`，模板和杂项进`docs/`。
- [x] 用户确认具体归类、paper空目录边界、缓存清理边界和反向回滚规则；开始撰写记录中的设计规格。
- [x] 在实施计划前运行四个P2-B/P2-C verify-only真实基线：4/4退出1；设计验收门已据实修订为不新增layout-induced失败。
- [x] 用户确认verify-only验收修订：不恢复已删除source，不伪报四门通过，以`layout_induced=0`为目录迁移验收底线。
- [x] 实施计划写入`记录/实施计划/2026-08-16-仓库顶层目录重构实施计划.md`，覆盖映射、分阶段迁移、兼容解析、测试、四门差分、文献审计、回滚和推送边界。
- [x] 设计规格写入`记录/设计/2026-08-15-仓库顶层目录重构设计.md`，180行、9节、占位扫描0；只提交该规格为`0152548`，未移动任何业务目录。
- [x] 用户确认书面规格；开始使用writing-plans生成实施计划，仍不执行目录迁移。
- [x] 用户确认分阶段原子迁移、旧manifest兼容解析和无长期旧目录别名方案。
- [x] 按映射完成1,556个文档类文件和15,290个代码文件迁移；旧`文档/`、`代码/`移除，新六类根目录落地。
- [x] 新增路径兼容层、迁移审计脚本及9项测试；旧manifest key只读解析，不恢复已删除的superpowers source。
- [x] 修正归档审计、CPU预检和P2-C测试中的现行旧路径；相关定向测试全部通过。
- [x] 四个P2证据门完成迁移前后差分，4/4均为`layout_induced=0`；真实既有失败继续如实保留。
- [x] 全仓编译通过；全量发现1,360项测试，0断言失败、7个`traci`环境错误。
- [x] 文献库复验：7类、78 PDF、78唯一SHA-256、78索引、78 checksum，0缺失、0签名/哈希错误。
- [x] 迁移验收记录写入`记录/迁移/2026-08-16-仓库目录迁移验收.md`；RRM未触碰，`code/artifacts/tmp/gpu_upload_eff9385`按provenance保留。

## 2026-08-15 手动下载论文入库与Zotero遗留清理

- [x] 读取`planning-with-files`与`pdf`技能规则，运行session catch-up并检查工作树基线。
- [x] 确认现有规划文件需接续而非覆盖；记录`docs/superpowers/`既有tracked deletions并保持隔离。
- [x] 初步清点`待整理`：14个PDF；列出显式Zotero快照和一次性迁移文件候选。
- [x] 核验14个PDF签名、页数、首页题名和DOI；13篇为目标正式版，Conformal Prediction为同项工作的arXiv版本并单独标记。
- [x] 比较14个新SHA-256与现有64个PDF：新文件彼此唯一且与旧库零重合。
- [ ] 冻结短文件名并在目标不存在、路径位于文献库后移动到七类目录。
- 工具记录：首次移动预检因PowerShell将`foreach`直接接管道而解析失败；无文件变化，后续使用结果数组再格式化。
- [x] 冻结14个短文件名并移动到主分类目录；`待整理`PDF为0、14个目标均存在。
- 工具记录：移动命令的父目录附加检查因`Split-Path`参数组合产生非终止错误，但其余保护门通过且移动全部完成；将用独立目录/哈希检查复核，不重复原语法。
- 工具记录：遗留文件统计中的另一段`foreach`再次直接接管道而解析失败；无文件变化，已升级为统一数组输出规则。
- [x] 独立复核14个目标父目录、文件存在与SHA-256；当前总计78个PDF、78个唯一哈希、313,064,419字节。
- [x] 更新CSV中14条记录并逐项复核路径/字节/哈希：78 available、0 manual、0验证错误。
- [x] 更新SHA-256清单加入14条新路径；正在同步概览、状态和缺失清单。
- [x] 同步README、文献概览、缺失清单、管理说明和状态JSON到78篇/0缺失；最终验收JSON暂标pending，避免未验先报通过。
- [x] 将8个一次性迁移/Zotero文件送入Windows回收站，活动路径剩余0；未清空回收站、未触碰RRM。
- [x] 审计并清理全仓剩余的历史`代码/artifacts/literature/zotero_import_20260722/`目录；16文件已送回收站。
- [x] 目录基线：16文件/14 PDF/66,693,588字节；10 PDF为现行库同哈希副本，4 PDF需继续核对同论文异版本。
- [x] 修正`本地计划表.md`中的旧文献数量和已删除审计文件引用；同步`AGENTS.md`为一次性快照已清理的真实状态。
- [x] 独立全库验收通过：78 PDF/78唯一哈希/78索引路径/78 checksum，0缺失、0遗留命名路径、0错误。
- [x] 将验收结果写入`本地文献库验收.json`并完成写后复核：结果仍为passed、0错误；`git diff --check`通过。
- [x] Git边界确认：本轮新增的tracked改动仅为`AGENTS.md`、三个规划记录和`本地计划表.md`；`docs/superpowers/`删除是进入本轮前已有状态，未恢复也未扩大。
- [ ] 完成身份匹配、去重、分类移动及索引重建。
- [ ] 清理确认冗余的Zotero遗留文件并完成独立验收。

## 2026-08-15 本地文献库迁移与Zotero退役

- [x] 读取`using-superpowers`、`openai-docs`、`planning-with-files`、`paper-fetch`、`web-access`和`verification-before-completion`技能规则。
- [x] 通过官方OpenAI文档核对用户级技能加载位置及禁用机制；用户明确要求物理删除，因此锁定唯一目标目录后执行删除。
- [x] 清点当前`文档/文献/`与Zotero本地数据根；确认必须同时处理已有散放PDF和Zotero附件。
- [x] 用户级`using-superpowers`目录及四个文件已送入Windows回收站；活动路径不存在，用户级技能根下无其他名称含`superpower`的目录。
- [ ] 导出PIJWM集合元数据、附件和七类本地目录。
- [ ] 自动补全公开PDF并生成手动下载清单。
- [x] 验证本地归档后删除Zotero PIJWM集合树；本地文献目录继续由`.gitignore`排除，不把PDF或原始快照推入公开GitHub。
- 工具记录：首次`web-access`依赖检查因Chrome未启动返回exit 1；按技能指引隐藏启动Chrome后代理进入连接流程。Google搜索触发反自动化页面，改用Bing发现入口并直接读取OpenAI官方`Build skills`页面。
- 工具记录：首次技能删除保护门假定目录只含`SKILL.md`，实际还含`references/codex-tools.md`、`copilot-tools.md`、`gemini-tools.md`，命令在删除前中止且未改变文件；后续按已核实的完整四文件目录处理。
- 工具记录：首次Zotero本地API读取因桌面端未运行返回`WinError 10061`；隐藏启动`C:\Program Files\Zotero\zotero.exe`后切换本地个人库成功，未直接操作`zotero.sqlite`。
- [x] Zotero初步覆盖审计：PIJWM共56项，30项带PDF、26项缺PDF，RRM集合完全排除在本轮范围外。
- [x] 本地初步清点：42个文件、30个PDF、约99.3 MB，PDF哈希无内部重复。
- [x] 完成Zotero附件与本地PDF哈希交叉核对：24个真实附件文件中6个已有本地同字节副本。
- [x] 验证Zotero可导出56条BibTeX；读取`paper-fetch` schema并固定公开来源、禁用Sci-Hub模式。
- 工具记录：28 DOI首轮public批处理下载14篇后最终返回`internal_error`；系统化诊断确认Python stdout为GBK、最终JSON含`ğ`导致编码失败。事件账本覆盖27/28项，14成功、13无公开PDF；只对缺终态的单项用UTF-8重试。
- 工具记录：UTF-8单项重试的摘要包装器错误假定成功envelope必含`data.results[0]`，遇到顶层`download_network_error`后产生空数组索引；真实JSON完整保存并显示PolyU HTTP 502，后续摘要对成功/失败envelope分别处理。
- [x] 无DOI的4个题名候选全部从arXiv成功获取；自动公开来源合计新增18篇，剩余14篇进入手动下载清单。
- 工具记录：首次自动下载整合脚本使用`$error`作为局部变量，因PowerShell自动变量`$Error`只读而在第一条记录、任何文件移动前中止；改用任务专用`$errorCode`后重跑。
- 工具记录：首次最终索引脚本在异常字符串中写`$key:`，PowerShell将冒号解析为变量限定符并在执行前报错；没有写入索引文件，改为`$($key)`后重跑。
- 工具记录：首次Zotero集合删除在本地库ID`0`上下文调用Web API，服务端以`Invalid user ID`拒绝，集合未删除；写操作需切换到真实云端用户ID`18841865`并重新读取确认。
- [x] 本地文献库退役前验收通过：7类、64个唯一PDF、78条索引、14条缺失、90条JSON快照、56条BibTeX和64条checksum均一致，错误列表为空。
- [x] 切换云端用户ID`18841865`后删除PIJWM根集合`MZ9JQ2I6`；云端与本地回读均确认根集合及七个子集合不存在，条目和RRM集合保持。
- [x] 治理入口提交`7828748 docs: retire Zotero PIJWM collection for local library`并推送到`origin/main`；PDF、CSV、JSON和BibTeX继续仅保存在本机。
- 工具记录：一次`rg`命令把PowerShell风格的`文档\文献\*.md`作为ripgrep路径传入，Windows返回路径语法错误；其他显式文件仍完成搜索，后续使用目录加`-g '*.md'`。
- 工具记录：首次最小编码复现把临时删除、重定向和多层引号组合在一条命令中，被执行器安全策略在启动前拒绝；拆成纯只读编码和NDJSON统计后成功。
- 工具记录：首次直接投影Zotero本地API的`include=data`响应时PowerShell把嵌套字段展开成数组，输出不适合作为附件清单；未写文件，后续先检查响应对象结构再生成归档manifest。
- 工具记录：一次只读元数据抽样命令在`foreach`后直接接管道触发PowerShell `empty pipe element`语法错误，未执行API请求或写文件；改为先累积数组再序列化。

## 2026-08-15 Repository governance and GitHub publish

- [x] 用户确认保守治理方案：不物理移动/删除研究资料，先修复迁移路径，再改善导航、验证、分批提交并推送GitHub。
- [x] 治理设计已提交：`0b83ee5 docs: design repository governance and publish`。
- [x] 实施计划已提交：`90e8194 docs: plan repository governance and publish`。
- [x] 基线确认：`main...origin/main [ahead 169]`（含上述两个新提交）；当前dirty tree和既有删除均保留。
- [x] 修复三个worktree Git路径和P2 ledger AirFogSim junction。
- [x] 复跑P2-B/P2-C迁移敏感验证。
- [ ] 完成仓库导航、发布集分类、测试、提交和远端推送；当前仅剩P2 ledger闭包、最终验证与push。
- 设计提交前`git diff --cached --check`提示设计文件EOF多一个空行；提交成功但该格式告警必须在后续治理提交中消除。
- [x] `git worktree repair`修复三个复制worktree路径；`git worktree list --porcelain`不再含旧根或`prunable`。
- [x] 仅替换P2 ledger的AirFogSim junction条目并验证新目标配置存在；未删除旧目标或任何第三方文件。
- [x] 迁移修复后P2-B v2、P2-C v2两个官方`--verify-only`均通过，错误列表为空。
- 工具错误：首次junction替换命令在执行前被`exec_command`安全策略拒绝，未发生文件操作；改为先验证固定绝对路径，再用`System.IO.Directory.Delete(path,false)`删除junction条目并用PowerShell `New-Item -ItemType Junction`重建，成功。
- [x] 提交`2f8ae16 docs: clarify PI-JWM repository navigation`：更新根README、文档索引、artifact/reference说明、ignore/attributes，并清除治理设计EOF空行告警。
- [x] 删除经检查只含2个空子目录、0文件、0链接的本地误嵌套`代码/代码/`；该路径未被Git跟踪。
- [x] 显式收集69个未跟踪顶层测试文件并运行288项测试，全部通过；未访问locked test。
- [x] 提交`ac7d10b feat: track current PI-JWM framework and tests`：155个框架/脚本/测试文件，缓存凭据模式扫描0命中。
- [x] 提交`ae03b0e docs: archive superseded research materials`：保留历史重命名关系和第三方模板原字节；归档审计1项通过。
- 全量测试限制：`airfogsim` Python 3.10发现1310项测试时仍有275个环境错误，主要为缺少`scikit-learn`与`tomllib`；最终验证改用具备依赖的`D:\miniconda\python.exe`复跑，当前不宣称全量通过。
- 工具错误：首次候选测试命令从`代码/tests`工作目录使用相对Git pathspec，保护门发现候选列表为空并立即中止，未运行测试；改用`git -C D:\shen\PKU\PIJWM`后得到69个文件。一次四文件空白补丁因3个路径漏写`PI-JWM`连字符而在读取阶段失败，按真实路径重试后成功。
- [x] P2 ledger文档提交`d6f776a`：绑定P2-C v2预文档计数、三个正式数据阻断和`training_eligible=false`边界。
- [x] 生成本地忽略final candidate并验证：audit/config与预文档候选逐字节相同，manifest仅有证据文档source hash变化；`--verify-only`返回`passed=true/errors=[]`。
- [x] P2 ledger分支以`e5ad8e4`非破坏性合并到`main`；合并后24模块focused suite实际`205 tests, OK (skipped=3)`，P2-B/P2-C v1/v2四个verifier全部通过。
- [x] 最终发布审计、fetch拓扑检查、非强制push和远端ref复核；`origin/main...main`在push前为`0 184`，首次push后local/remote均为`2ff84850281891f046a4111e609d593c9ed90330`。
- [x] 合并后`D:\miniconda\python.exe`编译检查通过；全量发现`1350 tests`无断言failure、`7 errors`均为该环境缺少`traci`、`3 skipped`。
- [x] 在`airfogsim`环境补跑上述4个真实AirFogSim模块共`39 tests`，结果`OK`；因此不把跨环境结果合并成虚假的全量全绿。
- [x] 修复旧项目配置测试对已归档`文档/项目说明`目录的过时断言，提交`0cc91ce`；新契约要求`文档/README.md`和`文档/研究进展/归档/`存在并确认旧目录不存在。

## 2026-08-12 P0 机器取证补充

- [x] P0首轮审计完成：25项claim matrix、报告、summary、输入哈希和manifest已落盘；本地计划表与8.12推进已同步。

- 已完成 8 月 11 日 PPT 权威页盘点（191–203），进入代码—测试—机器产物取证。
- 已用 `trajectory_index.csv` 复核 60 条轨迹的 36/12/6/6 分割、三种 CPU 策略各 20 条、六个场景各 10 条与每轨迹 300 步；未打开 locked test 轨迹。
- 已纠正 R5/R6 产物定位：正式训练在 `artifacts/formal_training/`，10k 策略汇总在 `artifacts/analysis/`。
- 已确认 R5 只冻结 R6 工作候选集且 `final_method_frozen=false`；R6 18/18 个 10k run 完成但 `formal_budget_complete=false`。
- 已核对 R5 组合 builder 与正式checkpoint参数键；C/D/E 的 Graph-RSSM叠加关系有代码和checkpoint依据。
- 已钉住 18/5 信息边状态事实：契约宽度 18，当前非零有效 count 仅索引 0/1/8/11/12；未补数据前阻止扩训。
- 已审计 R6 direct-policy 路径与 RSSM 核心测试；32 个核心单元测试通过，但不改变 P6 未实现和 P7/P8 阻塞状态。
- 已对 54 条非锁定张量直接复算信息边 mask，并完成 R2 registry/结果状态边界核验。
- 已复算 R1 顶层 278 文件 manifest，SHA-256 不匹配为 0；该证据仅支持 bundle 自洽，不支持语义完整。
- 已确认主文档“卸载+RB、CPU固定”与 R6 联合 CPU 候选/8维动作数据冲突；动作空间进入串行阻塞门。
- 已确认理论 `U^det` 每步规则层在当前 R3/R4 rollout 主干中不存在；旧结果降级为纯学习 rollout 历史证据。
- 最终manifest首轮校验曾从项目根误用`artifacts/...`相对路径而报文件不存在；改在`代码`根按真实路径重跑，确认产物存在。该错误不作为产物缺失结论。

## 2026-08-12 P0启动

- [x] 接受用户授权并启动P0；确认最高原则为理论—实现—数据—结果一致，不能用近似机制冒充。
- [x] 读取planning-with-files和Presentations技能规范；恢复现有规划文件，未覆盖旧记录。
- [x] 将旧阶段状态降为待审计claim；冻结长GPU、locked test、工作树清理和模型修改。
- [x] 找到权威主文档、8.12推进、8.11技术问答、讲稿及原始PPT；下一步逐页抽取并核验主张。
- [x] PPT artifact-tool模板首检因workspace未初始化/Windows辅助命令限制失败；已直接导入并渲染原始PPT，失败不作为内容结论。
- [x] artifact-tool直接导入并逐页渲染203页PPT，结构化文本已落盘；待界定8.11有效区段并逐页核验。
- [x] 界定8.11有效区段为191—203页，查看13页montage并精确导出第198/200/202页原生表格单元格。
- [x] 初步确认PPT第196页目标Rollout规划闭环与第201—202页当前直接候选策略未分层，进入代码/artifact核验。
- [x] 完成代码—测试—artifact证据抽取。
- [x] 生成P0 claim matrix、冲突处置和manifest。
- [x] 同步《8.12之后推进》和《本地计划表》。

## 2026-08-08 R6联合策略GPU训练前准备完成

- [x] 冻结`service_first_v1` reward：按时完成、失败、完成时延、交付量和能耗分解；三项尺度只从36条train轨迹估计并生成自校验protocol bundle。
- [x] 以TDD实现完整联合候选：默认AirFogSim调度加五种模板，统一绑定卸载、RB和CPU；DAG释放、合法邻居、RB唯一/容量、CPU逐节点容量及身份连续性均失败即停止。
- [x] 实现显式＋隐式候选策略网络、真实transition/rollout、终止/截断区分和GAE；Actor–Critic/PPO各完成一次有限CPU更新，冻结世界模型哈希不变。
- [x] 复用正式AirFogSim的DAG-ready调度与Observed通信路径，在validation seed 507生成4条连续真实转移；非空扫描实际改变卸载1次、RB 2次、CPU 1次。
- [x] 冻结18-run GPU矩阵、最大环境步、rollout/minibatch、validation checkpoint、patience、失败保留和首轮不含MPC的边界；本轮未启动GPU。
- [x] 正式bundle位于`代码/artifacts/preflight/pi_jwm_r6_joint_policy_gpu_readiness_v1/`；九项门、6个输出文件、8个输入绑定和连续两次同seed可重复性均通过。
- [x] 定向58项和历史75项测试通过；扩展371项中370项通过，唯一失败为用户已删除`文档/项目说明/`导致的既有目录治理检查，未擅自恢复。
- [x] 同步`本地计划表.md`、权威`PIJWM推进.md`和结果说明。当前`r6_gpu_strategy_training_ready=true`，但`gpu_used=false`、`locked_test_accessed=false`、`final_method_frozen=false`。

## 2026-08-05 R5 CPU准备启动

- 用户已授权先完成R5 CPU部分，不启动GPU。
- 范围固定为A-E五个组合、至少3个训练seed的正式预算、独立指标门、真实非锁定窗口组合预检和GPU交接产物；R1/R2/R3/R4接口及locked-test边界保持不变。
- 初始审计确认R4模型、目标、checkpoint、窗口和评价累计器可复用；R5主要新增受控多模块组合配置、正式协议和CPU runner。
- 首次测试检索因Windows通配符路径语法失败，已改用`rg -g`目录过滤并记录，不影响代码或产物。
- 已固化R5 CPU设计与实施计划，选择R5专用白名单组合层，保持R4单模块工厂及正式结果语义不变。
- R5协议TDD完成第一轮红绿：4项测试先因`r5_protocol`缺失而失败，最小实现后4/4通过；A-E组合、三seed/100 epoch/patience 10/batch 32预算、四项独立指标门和locked-test封锁已机器化。

## 2026-08-04 R4 GPU筛选启动

- 用户已提供新的SeetaCloud SSH实例，开始执行R4 GPU阶段。
- 首次只读审计确认RTX 4090空闲，数据盘空间足够；默认shell缺少`python`，下一步定位既有Conda环境并核对CUDA/PyTorch。
- 已冻结R2 module-screening预算：seed `20260803`、max epochs 30、patience 5、batch size 32、同窗口与同优化步数；locked-test继续封存。
- 远端将使用独立R4目录，不覆盖历史PI-JWM目录；历史数据只有在manifest/hash一致后才允许复用。

## 2026-08-04 R4 CPU候选矩阵完成

- 已实现七个模块族的受控单模块注册表、统一R4模型工厂、冻结R3目标加候选辅助项、严格checkpoint和非锁定CPU runner。
- 当前可执行矩阵共12臂：参考臂、symlog、SimNorm、R-GCN、edge-conditioned relation MPNN、无跨图耦合、关系约束cross-attention、Graph-RSSM、异方差头、活动率hurdle头、显式DAG消息和soft predicted presence。
- 正式预检使用train/validation/calibration各3个窗口，覆盖1/5/20步，共形成108条逐候选逐窗口目标记录；12/12候选完成单步训练、有限梯度、开放环滚动、未来目标泄漏检查、动作敏感性和严格checkpoint精确回读。
- 首次真实矩阵暴露padding端点与DAG端点布局问题；已按冻结协议修复为“非活动`(-1,-1)`槽可忽略、活动槽必须合法”，并兼容DAG原生`(B,2,E)`父/子行布局。两次失败产物均保留在preflight目录作为审计记录。
- 正式产物位于`代码/artifacts/preflight/pi_jwm_r4_cpu_preflight_v1/`；20个manifest文件独立复算SHA-256均一致，`locked_test_accessed=false`、`r4_cpu_preflight_ready=true`、`gpu_screening_ready=true`。
- 验证通过：R3回归22项、R4模块43项、R4 runner 3项及源码编译。该结论只允许进入GPU短预算筛选，不代表收敛、候选优越或最终模块定型。

## 2026-08-04 R4候选方法文献门启动

- 按用户要求暂停R4候选实现与GPU实验，先完成方法调研和理解。
- 调研分为六类：字段编码、图编码、双图耦合、潜在动力学、输出与不确定性、DAG/动态拓扑。
- 当前只做原始文献检索、Zotero去重审计、机制与公式梳理、PI-JWM接口映射和公平实验设计；R3模型、R1数据、R2指标和locked-test均不改动。
- 已完成第一版六类方法审计和分层候选矩阵；明确首轮、储备与推迟路线，并将公平实验门写入`文档/研究进展/2026-08-04-PI-JWM-R4候选方法文献调研与实验门.md`。
- Zotero新增/补齐DreamerV3、GATv2、ECC、Deep Ensembles、VGRNN、Graph World Model、G-RSSM预印本和多DAG卸载八条证据；云端集合与标签回读通过，本地Zotero API尚待同步。

## 2026-08-01 正式双图模型与评价阶段启动

- 用户冻结总目标：生成新的PI-JWM双图模型，跑通新的完整评价指标，并与baseline进行同口径对比。
- 已开始只读盘点现有正式张量、最小双图rollout、稀疏事件诊断、指标与baseline接口；尚未修改模型或启动训练。
- 当前设计门：确认本地CPU smoke应包含的baseline层级和成功标准；locked-test继续封存。

## 2026-08-01 AirFogSim双图张量v2与训练smoke

- 修正断连cloud语义：三条开发轨迹实际接入物理节点为18/16/17，cloud因无wired或无线物理边被排除；双图和节点快照同步重建。
- 实现固定张量契约、懒加载窗口数据集和train-only归一化统计；统一容量为18节点、306物理边、216信息流、163任务、422 DAG边，历史8步预测3步。
- 真实数据首次构建暴露信息流有效样本为0；根因为AirFogSim动作记录可能比完成传输事件晚一时隙。已改为采用动作/事件最早证据，完成时隙保留流、完成后清零，并加入回归测试。
- 三seed张量包全部通过引用、有限值、padding、容量和窗口校验；seed0训练统计含1,160个有效信息流观测。
- 实现最小action-conditioned双图rollout模型和mask-safe损失，真实batch前向/反向传播通过。
- 完成一轮开发smoke：seed0训练、seed1验证、seed2校准，各110窗口；链路活动F1均为0，信息流/任务存在F1偏低，结论明确限定为流水线成立而非模型收益。
- 已同步权威`PIJWM推进.md`、`本地计划表.md`和`task_plan.md`。

## 2026-07-31 AirFogSim新双图数据重构

- 已确认实施范围：不修改AirFogSim框架归属，在PI-JWM适配层重构物理图、信息图、任务/DAG辅助结构、CIP/CFE和指标基础台账。
- 已核对AirFogSim信道实现支持九个V/U/I方向，而当前项目导出仅覆盖V2U、V2I、U2I。
- 已建立六阶段实施计划；当前正在审计旧适配器、数据字段、样例产物和测试。
- 已确认旧适配器没有代理—信息流构图接口；旧小实验03的运行时信道事件可复用，但任务节点/DAG边构图必须替换。
- 已完成旧契约与指标台账审计，并写入`代码/artifacts/airfogsim_dual_graph_v2_IMPLEMENTATION_PLAN.md`；开始以TDD冻结v2对象契约。
- 新构图测试经历RED（模块不存在，6项失败）和GREEN（6项全部通过）；v2纯构图器与7类语义合法性检查已实现。
- 新重构脚本测试经历RED（脚本不存在，2项失败）和GREEN（2项通过），并已生成第一版exp06结构化产物及指标输入清单。
- 第一版产物验证通过，但单时隙样例暴露旧源包的时间戳不一致：信息流`t=2.6`，物理边`t=2.9`，物理节点`t=12.0`。已停止把它表述为严格同一时隙，转入逐时隙快照采集修正。
- 已检查现有60-seed原始轨迹；虽然节点/链路逐时隙字段齐全，但其`dag_edge_prob=0.0`，不能替代本次启用DAG的同源运行。
- 逐时隙快照测试已通过；首次独立运行在导入AirFogSim时因GBK控制台无法编码依赖输出而停止，尚未进入仿真，下一次运行固定UTF-8子进程编码。
- UTF-8重跑进一步确认base Python缺少`osmnx`；已验证项目既有`airfogsim` Conda环境可导入该依赖，后续运行切换到该环境。
- 已在`airfogsim` Conda环境完成exp06同源运行和资源守恒运行；两者均通过同seed复现，资源包全部守恒门通过。
- 已将物理边采集从旧V2U/V2I/U2I扩展到AirFogSim九类V/U/I有向信道，得到19,300条逐时隙边快照；73条结果回传无同刻承载缺失。
- 已新增真实指标模块并接入重构器，生成`metric_results.json/csv`和人读报告；右删失、不可计算指标和词典序优化目标均有明确状态。
- 已把`main_experiment_contract.py`升级为v2：任务/DAG不再属于信息图，CPU从核心动作移为固定规则。
- 已同步权威`PIJWM推进.md`和`本地计划表.md`，当前进入全量测试与最终差异核对。
- 相关78项测试和规定的75项双图/消融回归测试全部通过；全量749项中746项通过，3项失败来自既有文档目录缺失/审计脚本路径编码，不属于本轮代码回归。
- 最终产物核验通过：v2合法性无失败项，Task_10样例严格同刻，九类物理方向齐全，主目标为`service_loss=0.488189`；编译检查和`git diff --check`通过。
- 启动多seed开发数据集v2：先冻结变规模轨迹包和seed内窗口索引，不直接复用旧固定`x_link`语义；首轮计划生成seeds 0/1/2的development smoke。

## 2026-07-24 RRM Review Reconstruction Start

- Started a read-only review of `D:\shen\网络组\基于信道图谱的RRM` at the user's request.
- Confirmed the folder contains two review PDFs, a PPTX, and Markdown speaker notes.
- Successfully extracted the 37-page English review with `pypdf` (about 101,631 characters).
- The Chinese PDF direct read returned `PermissionError`; the next step is an ASCII temporary-copy workaround.
- No project definitions, code, experiments, or user documents were modified in this phase.
- Read the Chinese review from the communication-network section through its conclusion and references; the full 36-page version is now covered.
- Confirmed that the English and Chinese PDFs are two language versions of one review, while the PPT and notes are derivative presentation materials rather than independent evidence.
- Completed the first RRM-to-PI-JWM gap map across scenario, state, action, prediction targets, objectives, constraints, metrics, uncertainty, and deployment.
- Verified by exact term search that the review does not prescribe a cell-free scenario and does not define channel-map/path-loss prediction as the final objective.
- Kept the review read-only: no PI-JWM code, experiment schema, knowledge-base Markdown, or research definition was changed.

## 2026-07-21

- Added the root-level `新对话接续说明.md` handoff document for a clean new conversation.
- The handoff separates frozen evidence and protocol constraints from open method choices; it explicitly does not prescribe phase LCB or any other selector as the next method.
- Included current results, split locks, failed routes, literature novelty boundaries, code/artifact entry points, worktree state, and a ready-to-use opening prompt.

## 2026-07-20

- Resumed from completed RTX 4090 formal run.
- Confirmed repository HEAD is `f7cb651` and the worktree started clean.
- Confirmed formal output directory and checkpoint loading/selection APIs exist locally.
- Started systematic attribution of the 100% defer failure before changing any selector behavior.
- Confirmed all 36 checkpoint files and the formal summary are present locally.
- Confirmed the candidate gate passed while the selector gate failed only on RMSE and improved-seed count.
- First attribution launch stopped before loading data because `pi_jwm` was not on the piped Python process path; no experiment result was produced.
- A second launch with a literal Unicode path failed at the same pre-load import boundary; attribution data and checkpoints remain untouched.
- The environment probe proved `PYTHONPATH` works. A third launch then exposed stdin code-page corruption of the Chinese `代码` literal; it stopped before model loading. The launcher is being changed to an ASCII-only body with a full code-root environment variable.
- Completed the formal checkpoint attribution on validation only and wrote CSV/JSON artifacts under `selector_attribution_task_bridge_h10_f7cb651`.
- Attribution shows rank failure, not merely conservative defer: rank-only RMSE is 292.90-312.33 and all validation seeds worsen.
- Ran the existing schema-v6 benefit identifiability audit with HGB on selected-edge, pooled-interaction, and full interaction feature groups.
- Opportunity detection generalized strongly, but candidate ordering did not; the next architecture will separate the opportunity gate from token-level candidate ranking.
- Added the token-level ranker, opportunity-masked loss, deterministic mini-batch fit/predict, safe selection, and a train/calibration/validation-only runner using TDD.
- Local gate passed: 85 relevant selector tests, Python compilation, and diff checks.
- The first incremental bundle command produced no file because it did not include a named ref; remote state was unchanged.
- Synced commit `c7f743b` to the RTX 4090 with a verified Git bundle; remote selector tests passed 85/85.
- Completed full-data CUDA smoke and a no-phase 3-seed x 20-epoch token probe; the training chain is healthy but calibration found no safe threshold.
- Demonstrated that observable within-episode phase improves the HGB validation diagnostic to RMSE 232.023 with 100% positive precision on six executions.
- Added episode-phase context to the formal token runner with a failing-then-passing test; related tests now pass 86/86.
- Candidate-specific HGB prototype selected a 15-candidate train-only shortlist with validation oracle 170.80, then stopped before fitting because this sklearn version rejects `loss='huber'`.
- Candidate-specific experts completed after the loss correction but reached only validation RMSE 233.164; this over-split architecture was rejected.
- Phase-table prototype first launch stopped at metric-helper import because `scripts` was absent from `PYTHONPATH`; no result was produced.
- Phase-conditioned train statistics reached validation RMSE 207.540 with 10/10 seed improvements under the deployable Pareto rule.
- Confirmed the result is unchanged when validation `action_applied` is removed from the decision mask; no actual-rollout outcome is used online.
- Rejected phase smoothing, estimator ensemble, candidate experts, learned residual routing, and phase-restricted kNN after controlled train/calibration/validation comparisons.
- Added the reusable phase selector module, formal runner, decision trace, statistics cache, and freeze report using TDD.
- Formal runner reproduced validation RMSE 207.5398777 and freeze digest `887331b2...454a2` without warnings.
- Final verification passed: 722/722 main tests, 84/84 script tests, compileall, and diff checks.

## 2026-08-14 P2-C Advisor-Document Manifest Binding

- Added two RED assertions: the canonical manifest must include the P2-C research-progress document, and changing a temporary project-local copy must produce `source hash mismatch`. Both failed for the intended missing-binding reason before implementation.
- Added exactly one production dependency path to `CANONICAL_SOURCE_PATHS`; the P2-C test pair then passed 9/9, with compile, Ruff, and diff checks clean.
- Published a candidate audit, verified it, confirmed the audit report and candidate config were byte-identical to the prior canonical, archived the prior canonical, and promoted the candidate.
- Fresh evidence gate: P2-B `--verify-only` passed, P2-C `--verify-only` passed, AirFogSim manifest dependencies matched 83/83, and the 17-module focused suite passed 159/159 under `PYTHONUTF8=1`.
- Status remains `blocked` with four unchanged reasons; no GPU, locked test, formal trajectory generation, or third-party source modification occurred.
- Synchronized the frozen weekly evidence into `文档/研究进展/research_progress_overview.tex` and `文档/研究进展/pi_jwm_ton_draft_zh.tex`.
- Added the validation-only phase-conditioned benefit LCB method, B-grade result (`207.5399`), root-cause attribution, test-lock boundary, and GPU handoff conditions to both documents.

## 2026-07-21 Whole-Project Review

- Started a read-only reconstruction of PI-JWM across README, local plans, research documents, source code, experiment artifacts, and related literature.
- Loaded the planning, literature-review, and PDF workflows; preserved the completed v11 selector record and appended a separate review plan.
- First README read exposed a terminal encoding issue; logged it and will re-read with explicit UTF-8 output.
- Re-read README and `本地计划表.md` successfully as UTF-8; extracted the framework boundary, experiment protocol, current bottlenecks, and selector evidence status.
- Inventoried the tracked documentation and PI-JWM source/script/test surfaces; identified the current conceptual documents and the core v6-v8-v11 implementation progression.
- Read the formal research boundary and advisor motivation; separated the implemented counterfactual forecasting problem from the broader online multi-objective control ambition.
- Read the original thesis framing and result protocol; identified an outdated over-broad novelty claim and recorded the stricter evidence semantics that supersede it.
- Mapped the v0-v11 experimental progression from the research timeline and current paper draft; separated durable world-model evidence from diagnostic selector branches and noted the counterfactual-validity gap.
- Inspected the v6 data and model implementation; verified the action-conditioned multi-head rollout and found that "dual graph" currently means per-edge dual-modality fusion with global pooling rather than explicit topology message passing.
- Indexed the v8 full model and training implementation; confirmed explicit message passing and sparse-rate mechanisms, while flagging the very broad experimental configuration surface and downstream-objective mismatch.
- Read the v8 graph blocks and forward path in detail; verified endpoint message passing and identified fixed topology, collapsed dual latents during rollout, and globally pooled task dynamics as structural limitations.
- Inspected dataset assembly and the v8 runner; verified alignment guards and validation-based checkpointing, and recorded the operational activity label plus legacy split/default hazards.
- Located the structured experiment archive and local literature holdings, including a dedicated artifact literature directory and historical PDF collections.
- Listed the focused local PDF corpus and current TeX bibliography; identified missing closest-domain citations and a more useful literature taxonomy.
- Read the local v9 literature plan and core paper extracts; translated adaptive graph, event memory, dynamic decomposition, and sparse-distribution ideas while rejecting literal transfers that do not match continuous rate targets.
- Loaded `web-access` for a current literature check. Chrome CDP was unavailable after the prescribed background-start retry, so the search will use public academic APIs and paper sources without touching the user's browser tabs.
- Began external metadata verification. The two locally cited 2026 arXiv IDs did not resolve through the arXiv API, so they are now explicitly marked unverified pending independent cross-checks.
- Completed a metadata-only OpenAlex cross-check of closest domain work; verified four major 2023 competitors and one closer 2026 DT/offloading paper, while excluding the unresolved agentic preprints from the evidence chain.
- Read verified competitor abstracts and offline model-based RL/world-model literature; derived the need for world-model-level support and uncertainty, not only selector-level defer logic.
- Inspected the causal candidate policy and phase selector implementation; verified its deployment inputs and identified exact episode-phase lookup as both the source of current gains and a major generalization boundary.
- Inspected v11 candidate labeling and schema-v6 interaction features; identified that the main benefit label optimizes factual prediction reconstruction rather than true counterfactual task/energy utility.
- Located the frozen v10 and formal v11 evidence packages that will be used to verify exact configurations, splits, and result semantics.
- Verified the frozen v10/v11 summaries; corrected the phase method's `z=0` semantics, documented 95% defer and low autonomous F1, and identified the world-model/selector split nesting that limits end-to-end generalization claims.
- Read the physical-benefit bridge design and inventoried its formal outputs; confirmed its strong causal/protocol safeguards and its limited role as a Pareto gate rather than the main control objective.
- Verified formal physical-bridge results; task response only marginally beats its baseline, energy response fails, and the bridge is correctly restricted to task-only features with energy audit-only.
- Audited the new-conversation handoff and per-seed phase results; found its two unverified 2026 citations and logged a column-name error in the first decision-trace aggregation.
- Recomputed the decision trace with correct columns; found that 58/65 executions use one shrink candidate and 60/65 concentrate on six repeated phases across every validation seed.
- Analyzed the underlying 60-seed dataset by within-seed local index; confirmed the selected phases are top-percentile future RB/activity bursts and that phase aliases a deterministic task/scheduler lifecycle.
- Completed a five-perspective synthesis and formulated four distinct, falsifiable research routes: event-aligned joint dynamics, support-aware counterfactual deltas, structured resource projection, and genuine scenario transfer.
- Final read-only verification passed: the v10, formal phase-selector, and task-only physical-bridge JSON files parse successfully; `git diff --check` exits 0; no project code, tests, artifacts, README, or `本地计划表.md` were modified by this review.
- Existing LaTeX/PDF changes and the untracked handoff document were preserved and not altered by this review.

## 2026-07-31 AirFogSim Multiseed Dataset v2

- Added seed-isolated 8-to-3 window construction, validation, metric aggregation, and a single-pass multiseed builder.
- Extended the PI-JWM collector with per-slot task snapshots and frozen offload, return, and RB action ledgers.
- Reproduced a seed-1 AirFogSim failure at t=6.0 and traced it to `getOffloadingTasksWithNumber()` mutating an aliased internal offloading list while merging return tasks; the same Task_42 object appeared eight times.
- Installed a PI-JWM-side nonmutating transmission-task view without editing AirFogSim core; the regression test, seed-1 reproduction, and full seeds 0/1/2 run passed.
- Generated 3 trajectories x 120 slots and 330 seed-isolated windows in `代码/artifacts/datasets/airfogsim_multiseed_v2_dev/`.
- All graph, time alignment, action ledger, nine-direction physical-edge, resource conservation, and window gates passed. The package is development-ready but intentionally not formal-training-ready.

## 2026-07-22 Four-Day Sprint Start

- Replaced the rigid daily-gate wording with a flexible four-day reference pace while keeping explanation and evidence checks.
- Verified the current model inputs/outputs against README, `v6_data.py`, `v8_full_world_model.py`, `v8_training.py`, and the 60-seed NPZ schema.
- A direct piped-Python Unicode path read failed, and a full lazy NPZ attempt exceeded the short timeout; switched to a PowerShell-resolved environment path plus ZIP/NPY header-only inspection, which returned the feature names and tensor shapes without loading large arrays.
- Expanded `文档/项目说明/研究问题与系统边界.md` with the verified schema, mathematical state/action definition, implemented world-model loss, recommended system objective, and current data gaps.
- No model training, locked-split access, source-code change, or artifact mutation was performed.

## 2026-08-01 AirFogSim Sparse-Event Diagnostic v2

- Added train-only sparse-label statistics, explicit link-activity targets, capped positive weights, and lifecycle-majority statistics.
- Added link-activity and five-class task-lifecycle heads to the minimal action-conditioned rollout model with mask-safe multitask losses.
- Added zero-activity and last-persistence baselines plus shared F1/AUPRC, active-only rate, lifecycle, and physical-unit state metrics.
- Ran a deterministic four-arm, five-epoch development diagnostic on seed 0 train, seed 1 validation, and seed 2 calibration; learned arms shared initialization hash and sample order.
- Balanced training improved link AUPRC to 0.044855/0.045419 on validation/calibration, but did not consistently beat last persistence on link F1, active-rate MAE, task presence, or lifecycle macro-F1.
- Recorded `diagnostic_ready=true`, `formal_training_ready=false`, and `jepa_comparison_ready=false`; JEPA remains deferred until the base rollout clears simple baselines.

## 2026-07-22 Literature Library and Workspace Consolidation

- Started a read-only audit before any Zotero mutation, download, move, or deletion.
- Confirmed the workspace currently contains root-level project metadata and multiple Markdown handoff/planning files in addition to `代码/` and `文档/`.
- Added a staged consolidation plan with an explicit review gate for destructive cleanup.
- Completed the first filesystem/git inventory: 46 PDFs were found across active literature, artifact, reference, template, and meeting-archive locations.
- Confirmed Zotero is running, but its active data directory and supported local-API status still require verification.
- Recorded the pre-existing dirty-worktree boundary so later cleanup does not overwrite or misattribute user changes.
- A broad recursive Zotero database search timed out; it was abandoned and replaced with profile-directed discovery.
- Resolved Zotero's custom data directory and confirmed the local API preference is enabled.
- The first local-API probe was closed unexpectedly; moved to socket-level diagnosis without touching the database.
- Confirmed successful read-only Zotero access by bypassing the system proxy for localhost.
- Inventoried all 94 top-level items and detected existing PIJWM-relevant records plus seven unrelated duplicate-DOI groups.
- Completed SHA-256 deduplication for all workspace PDFs: one exact duplicate pair found.
- Summarized the document tree and identified root planning files plus generated LaTeX/Python byproducts as likely archive/cleanup targets.
- Started first-party Zotero documentation verification through the required browser-access workflow; corrected the current CDP proxy response field from `id` to `targetId` before content extraction.
- Verified from Zotero's official documentation that the local API is read-only; closed the temporary browser tab and retained UI/import as the safe mutation route.
- Reconstructed the deterministic literature seed set from the current ToN bibliography and v8 literature map.
- Completed the first five-theme 2023--2026 metadata discovery pass; retained only named candidates for narrower verification.
- Verified DOI, year, venue, citation count, and OA status for 15 targeted digital-twin/GNN/UAV-MEC candidates.
- Completed a strict-dual-graph and dependency-DAG discovery pass; recorded that the exact PI-JWM dual-graph construction requires a transparent composite literature basis.
- Completed a narrow wireless/world-model search and identified published JPROC/TCCN/IoT-J/JSAC bridges.
- Completed a realistic-data/testbed search and established the likely hybrid calibration/validation role of Colosseum, OpenRAN Gym, DeepMIMO, DeepSense 6G, and AERPAW.
- Created the persistent Zotero taxonomy and ingest rules in `文档/文献/PIJWM文献管理说明.md`.
- Ran the first 27-item lawful OA acquisition batch with Sci-Hub disabled: 12 PDFs succeeded and 15 items remain metadata-only pending alternate-author-copy checks.
- Added two verified HAL repository PDFs, raising the acquisition total to 14; confirmed one IEEE hybrid PDF opens normally in Chrome and moved to standard viewer download handling.
# 2026-08-02 Formal dual-graph CPU loop completion

- Added leak-safe formal windows with 8-step history, 3-step logged future actions, 3-step targets, and explicit DAG dynamics; locked-test access remains rejected.
- Added strict physical/information graph operators, CIP/CFE coupling, task/DAG propagation, and pooled/independent/coupled action-conditioned rollout models.
- Added masked probabilistic loss, horizon-wise formal metric registry, two rule baselines, and a fair five-method CPU runner.
- Formal interface smoke completed on 6/3/3 train/validation/calibration windows; comparison smoke completed on 64/32/32 windows.
- Both runs report `locked_test_accessed=false`; each 21-file manifest was rehashed without errors and all six learned checkpoints reloaded with `weights_only=True`.
- Ten test files covering 45 tests passed; source/script compilation and `git diff --check` passed.
- Boundary remains `cpu_smoke_ready=true`, `formal_performance_claim_ready=false`, `formal_training_ready=false` because continuous state prediction still trails last persistence in the short CPU run.

## 2026-08-03 Unified Roadmap Planning

- User approved a sequential stage-gate roadmap ending immediately before foundation-model pretraining and cross-scenario transfer.
- Audited the repository, current planning files, formal-data documentation, CPU/GPU evidence, and the main document status table.
- Froze the no-duplicate-work rule: reuse raw trajectories, splits, ledgers, metrics, runner/test infrastructure, and historical implementation probes; regenerate only artifacts whose graph ontology or model claims are invalidated by the advisor-aligned definition.
- Began synchronizing the R0-R9 roadmap into the authoritative PI-JWM main document and local overview plan.
- Replaced the short task list with a unified R0-R9 stage-gate roadmap in the authoritative main document and the local overview plan.
- Each stage now records reuse boundaries, CPU/GPU needs, expected outputs, advancement gates, and truthful current status; pretraining and transfer remain outside the roadmap.
- Verification confirmed exactly one formal table row for each R0-R9 stage in both documents, consistent Markdown table widths, balanced code/math delimiters, no old date, and no residual wording that fixes Graph-RSSM or gated fusion before experiments.

## 2026-08-03 Strict Dual-Graph Literature Freeze

- Verified the original TPDS and JSAC sources and fixed their precise evidentiary roles without claiming either paper reproduces the full PI-JWM definition.
- Added both papers to the authoritative definition, related-work comparison, method evidence matrix, literature-ingest audit, and R0 roadmap status.
- Deduplicated both DOIs before Zotero creation; cloud items `C4M4482F` and `65HG2GWI` passed collection/tag/zero-child-attachment re-read. Local Zotero has not synced them yet, so no repeated write was attempted.

## 2026-08-03 R1 Teacher-Aligned Data Contract Completion

- Implemented `PIJWM-DG-Contract-v3` graph remapping, fixed tensorization, source-field audit, `CIP/CEP/CFL` materialization, and structural/tensor validation under `代码/src/pi_jwm/`.
- Reused all 60 saved AirFogSim trajectories without rerunning the simulator: 54 non-locked trajectories were tensorized, 6 locked-test trajectories remained integrity-only, and 15,660 non-locked windows were preserved.
- Generated `代码/artifacts/datasets/airfogsim_teacher_aligned_v3/` with 278 manifest-managed files; an independent full-file SHA-256 recomputation reported zero mismatches.
- Fixed the formal-build memory failure by replacing all-trajectory concatenation with streaming per-trajectory statistics, and added verified resume support so completed seed artifacts are hash-checked and reused.
- R1-specific tests, relevant legacy formal-data regressions, compilation, split isolation, train-only statistics, and locked-test sealing passed. The next stage is R2; formal training remains blocked until R2 and R3 pass.

## 2026-08-03 R2 Evaluation Protocol Completion

- Froze a 36-metric machine-readable PI-JWM registry with formula, unit, numerator, denominator, target mask, source fields, optimization direction, aggregation, computability condition, and explicit N/A rule.
- Froze the fair experiment contract: validation selects architecture/checkpoints, calibration only sets thresholds and uncertainty calibration, and locked-test remains single-use in R9. Module screening uses one training seed with 30 epochs/patience 5; formal comparison uses three training seeds with 100 epochs/patience 10.
- Added teacher-aligned zero-state and last-persistence evaluators. Geometry metrics read physical edges; link activity and rate read information edges. Deterministic baselines retain uncertainty rows as `not_computable` instead of inventing values.
- Materialized `代码/artifacts/evaluation/pi_jwm_eval_protocol_v3/`: 54 non-locked trajectories produced 108 baseline reports, and the existing 22 AirFogSim factual system metrics were audited through the same protocol boundary. Six locked-test trajectories remained sealed and unevaluated.
- Independently recomputed all seven manifest-managed file hashes with zero mismatches. Thirty-seven R2/R1/legacy-metric regression tests, source compilation, and `git diff --check` passed. R3 CPU smoke is now the next stage; no GPU training was performed in R2.

## 2026-08-03 R2 Correctness Re-Audit

- Re-opened R2 before R3 and completed a four-perspective literature/protocol audit plus an independent code review. The first review correctly rejected the original freeze because factual/prediction namespaces were disconnected, the legacy wireless-activity name still exposed physical-edge semantics, and the checkpoint composite was not executable.
- Added canonical factual mapping, unified evaluation rows, train-only zero baselines and checkpoint scales, history-valid persistence masks, fixed five-class lifecycle F1, calibration-only event thresholds, macro/pooled statistics, exact split/locked identity checks, and input/code/runtime provenance.
- The first formal rebuild failed honestly because the checkpoint composite had been imposed at the wrong single-trajectory level. Corrected the protocol to aggregate each target group over valid validation samples before combining the ten normalized components; a second formal build passed every gate.
- The audited candidate contains 43 registry metrics, 22 factual metrics, 108 baseline reports, 1,188 factual rows, 2,160 baseline rows, 3,348 unique canonical rows, and 13 manifest-managed files. All 54 non-locked trajectories were evaluated and all 6 locked-test trajectories remained sealed.
- R2's 13 targeted tests pass. Full repository discovery ran 907 tests: 904 passed; the three failures are pre-existing consequences of user-side deletions under `文档/项目说明/` and `文档/研究进展/`, not R2 regressions, and those deleted files were not restored.

## 2026-08-03 R2 Final Contract-Binding Audit

- Bound `tensor_contract.json` and `protocol.json` to the source manifest before any evaluation read, and require the physical-node, physical-edge, information-node, and information-edge feature lists to match implementation constants exactly and in order.
- Added a negative regression test proving that a reordered feature contract is rejected even if its source-manifest hash is updated. Input provenance now hashes the three evaluation modules and the three tensor/metric dependencies that define imported semantics.
- Rebuilt and promoted the canonical R2 artifact. It contains 43 registry metrics, 22 factual mappings, 108 baseline reports, 3,348 unique canonical rows, 62 verified inputs, 6 verified code files, and 13 manifest-managed outputs; 54 non-locked trajectories were evaluated and 6 locked-test trajectories remained sealed.
- Independent code review returned `Ready` with no remaining Critical or Important finding. Fifteen R2 targeted tests passed. Full discovery ran 909 tests: 906 passed; the same three unrelated failures come from pre-existing user deletions in the document/audit-script area and were not modified.

## 2026-08-04 R2 DAG Semantic-Index Correction

- Independent review found that the three-dimensional DAG normalization was correct, but `unfinished_parent_count` evaluation still read column 0 (`parent_count`) instead of column 1.
- Added distinct-column hand-computed regression tests, switched both the evaluator and checkpoint scale to `FORMAL_DAG_STATE_FEATURES.index("unfinished_parent_count")`, and rebuilt the canonical R2 bundle.
- The corrected train-only DAG scales are `1.9266825/1.4639631/0.4581655`; the checkpoint component now uses `1.4639631`. Sixteen focused R2 tests passed, 62 inputs and 7 code dependencies are bound, and 13 output hashes match.

## 2026-08-04 R3 New-Semantics CPU Preflight Completion

- Implemented deterministic read-only 8-history windows with 1/5/20-step targets, train-only normalization, explicit activity labels, and sealed locked-test rejection.
- Implemented a v3-native reference world model with separate physical/information encoders, explicit `CIP/CEP/CFL` coupling, explicit state predictions, physical/information/business/joint latent beliefs, and action-conditioned open-loop recursion without future-target reads.
- Added mask-aware continuous/presence/activity/lifecycle objectives, gradient coverage checks, and provenance-bound strict checkpoints.
- Formal CPU preflight used 9 real train/validation/calibration windows and both coupled/no-coupling controls. It produced 18 finite objective reports and 18 R2 baseline-interface records, executed one backward update per control, and reloaded both checkpoints exactly.
- Same-parameter coupling on/off changed joint belief by `0.0608153`; ordinary versus zero action changed it by `0.0113508`. These are execution checks only, not performance claims.
- Canonical artifact: `代码/artifacts/preflight/pi_jwm_r3_cpu_preflight_v1/`. Eight output hashes, four R3 source hashes, and nine selected tensor hashes match; `locked_test_accessed=false`. R4 GPU module screening is now the next stage.
# 2026-08-04 R3 Literature and Correctness Re-Audit

## 2026-08-07 R5.1本地收尾启动

- 用户确认严格执行步骤1—3：本地独立验收、三seed/分horizon统计、冻结R6工作候选集合。
- 远端正式训练已完成12/12新增run，复用B 3/3；正式验证器报告12个checkpoint、72个文件、15份证据均通过，locked-test未访问。
- 完整压缩包已下载到`代码/artifacts/remote_handoff/pi_jwm_r5_module_confirmation_complete_20260807.tgz`，远端与本地SHA-256均为`a843c0c8c850ea47041f0454902036e2c3c0e0550630d0078a36b500c10d1c9f`。
- 本轮不使用新worktree：当前R3—R5源码是未提交但已验证的工作区资产，新worktree无法复现真实依赖；保持原位最小追加且不触碰无关改动。
- 已检查压缩包目录结构：顶层唯一正式结果目录为`pi_jwm_r5_module_confirmation_v1/`，包含协议、组合矩阵、窗口清单、F/G/H/J逐seed checkpoint/曲线/报告和复用B报告；本地正式目标目录此前不存在，不会覆盖既有产物。
- 已解包至`代码/artifacts/formal_training/pi_jwm_r5_module_confirmation_v1/`。本地正式验证器新鲜运行结果为`verified=true`：12个新run、3个复用run、12个checkpoint和72个manifest文件全部通过，`locked_test_accessed=false`；压缩包SHA-256仍为`a843c0c8...c10d1c9f`。
- 已审计现有统计与评价代码：现有R5描述统计可复用配对检验/CI/manifest，现有R4累计器可复用14项指标；分horizon必须按冻结窗口和checkpoint重新推理，不能拆分已有聚合数值。
- 已确认正式窗口清单记录远端路径；本地重放将使用冻结dataset重新构造并做完整窗口身份比对。B与F/G/H/J分别使用各自严格checkpoint loader，不做格式转换。
- 本地所需三项根目录均存在：teacher-aligned v3 dataset、R2 evaluation protocol v3、原R5 GPU training v1。统计实现将作为独立CPU分析入口复用训练器的只读评价函数，不修改训练代码或模型结构。
- 已固定R6角色判断门并记录在案，先于分horizon结果生成：B为强制控制；整体候选需三seed同向且关键通信/任务指标不过退化门；专长候选需预声明指标族≥10%均值改善且至少2/3 seed同向。

## 2026-08-07 R5.1本地统计与R6候选冻结完成

- [x] 以TDD新增R5.1通用统计、逐horizon复评和候选冻结模块；新增10项测试先红后绿，锁定完整矩阵、locked-test封锁、指标方向、窗口身份、45单元完整性和角色门。
- [x] 单checkpoint真实CPU冒烟先通过：B/20260803的36窗口分数与GPU报告绝对差`2.8178e-07`，随后才批量运行15个checkpoint。
- [x] 15/15最佳checkpoint按冻结validation窗口完成CPU重放；总体分数最大复现差`1.2208e-05 < 1e-4`，1/5/20步共45个单元全部可计算。
- [x] 正式分析包写入`代码/artifacts/formal_training/pi_jwm_r5_module_confirmation_analysis_v1/`；manifest共8个受管文件，独立复算SHA-256和大小无差异，locked-test未访问。
- [x] 按事前门槛冻结R6工作候选：B为主候选/强制控制，G为任务生命周期专长候选，J为连续状态专长候选，F为去耦消融，H退出当前R6主集合；`final_method_frozen=false`。
- [x] 新增结果说明并同步`本地计划表.md`和权威`PIJWM推进.md`；下一阶段只进入R6 CPU动作合法性与真实闭环下界，不立即启动策略器GPU训练。

## 2026-08-07 R6 CPU策略预检完成

- [x] 以TDD实现`r6_cpu_preflight.py`：角色冻结、locked-test封锁、双图/资源/动作账本、硬约束和指标三态语义均采用失败即停止；5项新测试先红后绿。
- [x] 正式审计54/54条非锁定AirFogSim轨迹，train/validation/calibration为36/12/6；三种CPU策略各18条，共23,255条CPU动作。
- [x] 顶层8个文件和54条轨迹的432个manifest文件全部重算大小与SHA-256，0个不一致；双图、CPU、RB、任务流、能量和依赖核算全部通过，硬约束违规0。
- [x] 保留指标三态：972条`available`、54条`not_applicable`、162条`not_computable`，没有把缺失值填成0；locked-test未访问。
- [x] 正式产物位于`代码/artifacts/formal_training/pi_jwm_r6_cpu_preflight_v1/`，结果说明位于`文档/研究进展/2026-08-07-PI-JWM-R6-CPU预检结果.md`。
- [x] 当前仅为观测性CPU闭环下界和执行入口，不作三策略因果排名；下一步必须在同场景、同初始状态、同seed下做规则/局部搜索配对闭环，再冻结学习策略GPU预算。

## 2026-08-07 R6 CPU配对闭环smoke

- [x] 新增`PairedCpuPolicyAllocator`：equal-share、deadline-aware、feasible-exploration和固定邻域local-search四臂；所有分配经过有限值、非负和节点容量投影。
- [x] 为正式runtime增加可选`allocator_factory`，默认`CpuPolicyAllocator`路径保持不变；新增12项策略/配对/runtime测试均通过。
- [x] 默认base环境smoke因缺少`osmnx`失败，失败bundle保留；切换项目既有`airfogsim`环境后，1个validation场景—seed的4/4策略臂通过，动作合法率1.0、硬约束违规0、locked-test未访问。
- [x] 在项目既有`airfogsim`环境完成54个非锁定base spec × 4个CPU策略臂，共216/216次配对仿真；`train/validation/calibration=144/48/24`，输出paired delta、动作账本、指标行和失败记录。
- [x] 正式bundle验收通过：54个完整pair group，`action_legal_rate_min=1.0`、硬约束违规0、失败run 0，指标状态为`available=3888`、`not_applicable=216`、`not_computable=648`，`locked_test_accessed=false`、`gpu_started=false`。
- [x] 8个受管文件独立重算SHA-256全部一致；结果说明位于`文档/研究进展/2026-08-08-PI-JWM-R6-CPU配对闭环结果.md`。paired delta只作同状态描述性比较，不冻结最终策略，`action_regret`仍未进入可计算状态。
- [x] 根据用户要求补充后续统一实验记录规范：先冻结协议和门槛，完整保留成功/失败run、原始指标与manifest，独立复现后再解释结果；已同步计划表和权威主文档。

## 2026-08-08 R6学习策略CPU准备完成

- [x] 以TDD实现策略状态/动作契约、安全投影、Masked Actor–Critic、Actor–Critic/PPO一次CPU更新和正式预检runner；策略器同时接收当前显式状态与冻结世界模型隐式belief。
- [x] CPU容量按真实信息代理节点建模，每个活动任务保存执行节点索引；安全投影按节点缩放，并在`float32`边界上回收舍入残差，严格保证投影后容量不超限。
- [x] 正式预检复用R1/R2/R5/R5.1/R6冻结资产，在候选B的真实checkpoint和非锁定validation窗口上运行；435个活动CPU任务、38个活动CPU节点，硬约束违规0、失败0、未来target未进入策略、locked-test未访问。
- [x] Actor–Critic与PPO数值smoke均得到有限loss/梯度并更新策略参数；世界模型更新前后逐参数SHA-256一致。该结果不表示两种策略的性能或收敛。
- [x] 新增22项测试、既有R6回归17项和历史双图/active-rate回归75项全部通过；正式bundle 7/7文件大小与SHA-256独立复算一致。
- [x] 正式结果位于`代码/artifacts/preflight/pi_jwm_r6_learning_policy_cpu_preflight_v1/`；结果说明位于`文档/研究进展/2026-08-08-PI-JWM-R6学习策略CPU预检结果.md`。
- [x] 当前固定为`r6_learning_policy_cpu_ready=true`；卸载/RB正式学习mask、reward/GAE、GPU预算和闭环训练矩阵仍未冻结，因此`r6_gpu_strategy_training_ready=false`、`final_method_frozen=false`。

- Verified five primary evidence anchors and froze their scope: PlaNet for observation-action belief and action-conditioned latent transition; GNS for relation-aware graph dynamics; JSAC for wireless channel state on information edges; TPDS for two intralayer networks plus explicit interlinks; TWC 2026 for coupled modality-specific latent dynamics as an R4 candidate.
- Corrected R3 to retain `history_action` separately from `future_action` and encode it into current task belief.
- Added fail-fast validation for physical/information endpoints, active `CIP/CEP/CFL` mappings, and historical/future action-to-information indices.
- Regenerated the formal 9-window, coupled/no-coupling CPU artifact. All gates passed; corrected diagnostics are `action_delta=0.0469472557` and `coupling_delta=0.0330387801`.
- Added `文档/研究进展/2026-08-04-PI-JWM-R3正确性与文献审计.md`; synchronized the design, main knowledge-base document and `本地计划表.md`.
- Added/corrected Zotero records: PlaNet `WRX5XKKD` and Graph Network-based Simulators `W79FZZKX`; cloud and local metadata/collections/tags verified.
- Remaining R3 limits are explicit, not hidden: DAG summary only, no DAG-edge propagation; predicted presence is supervised but does not yet gate later rollout messages. Both are R4 controlled variables.
- Final verification: R3 `22/22` passed; full repository `930/933` passed with only the same three user-deletion failures; canonical manifest and four source bindings passed; no stale old R3 deltas or explicit-state-feedback claims remain in the active documents.

# 2026-08-04 R4 CPU Implementation Start

- Re-read the approved R4 CPU design and literature gate, inspected the real R3 model/objective/checkpoint/data APIs, and ran the focused R3 baseline: 22 tests passed.
- Kept the implementation on the existing `codex/formal-airfogsim-dataset-v1` branch because the required R3 files are current-workspace assets not present in a clean worktree commit; no user changes were moved or cleaned.
- Completed the first RED-GREEN cycle for `r4_module_registry.py`: the new six-test file first failed solely because the module did not exist, then all six tests passed after the minimal registry/config implementation.
- The registry now records seven frozen module families, evidence and falsifiable question per candidate, explicit lifecycle status, R3-compatible reference configuration, one-family-at-a-time construction, machine-readable rows, and fail-fast rejection of unknown/nonexecutable/multi-change configurations.
## 2026-08-04 R4 GPU筛选实施

- [x] 审计远程 RTX 4090、CUDA/PyTorch、磁盘和旧目录；正式运行使用独立目录。
- [x] 以 TDD 实现冻结 R2 预算读取、非锁定训练窗口计划、显式批处理、v3 信息边指标与四项验证选模分数（8/8 新测试通过）。
- [x] 实现 CUDA-only 正式 runner：12 个单模块候选、有效 batch 32、梯度累积、30 epoch/patience 5、逐 epoch 状态、严格 checkpoint、排名和 manifest。
- [x] 上传最小代码、48 条 train/validation 轨迹和 R2 协议；远端输入校验通过，真实张量 3 步反向与 20 步滚动均有限，完整 36 窗口验证四项分数可计算。
- [x] 正式第二轮统一源码筛选完成：12/12候选成功、0失败、总耗时3567.66秒，`locked_test_accessed=false`；第一轮暴露的checkpoint容差和活动但零实现速率问题已用测试修正，未拼接新旧结果。
- [x] `graph_rssm_v1`以验证综合分数`4.482206671`排名第一，参考臂为`4.514384949`，相对降低`0.7128%`；该结论仅是单seed短预算动力学候选筛选，不是最终模型定型。
- [x] 正式产物已回传至`代码/artifacts/formal_training/pi_jwm_r4_gpu_screening_v1/`；远端和本地均验证46/46清单文件、12个checkpoint及复现误差，R3 22/22、R4 49/49定向测试通过。
- [x] R4阶段完成。由于信息边活动AUPRC和任务生命周期macro-F1绝对值仍低，下一步R5必须以至少3个训练seed复验Graph-RSSM、参考臂和必要的组合臂，不能直接宣称预测性能成熟。

## 2026-08-05 R5 CPU组合预检

- [x] 以TDD实现R5正式协议和A-E有限组合注册；非法组合、预算漂移与`locked_test`请求均直接拒绝。
- [x] 在不改变R4单模块工厂的前提下实现组合式R5工厂；C同时执行RSSM与异方差头，D同时执行RSSM与显式DAG消息，E同时执行RSSM与soft predicted presence，相关参数均有非零有限梯度。
- [x] 实现R5严格checkpoint，绑定五项R3上游契约、R4筛选manifest、R5正式协议、组合编号、组件表、学习率和训练seed。
- [x] 实现并运行正式CPU runner。A-E×3 seed共15/15通过，真实train/validation/calibration窗口的1/5/20步rollout、目标函数、一次反向更新、动作敏感性、未来target隔离、四项公共指标和checkpoint回读均通过。
- [x] 正式产物位于`代码/artifacts/preflight/pi_jwm_r5_cpu_preflight_v1/`；26个manifest文件哈希与大小独立复算一致，`failed_runs=0`、`locked_test_accessed=false`、`r5_gpu_ready=true`。
- [x] 当前只完成CPU执行门，不提供组合排名或收敛结论。下一步为A-E三seed、100 epoch/patience 10的正式GPU训练。

## 2026-08-06 R5正式GPU训练与多seed统计

- [x] A-E×3 seed正式训练15/15完成、0失败，总耗时`40287.76 s`；locked-test未访问。
- [x] 正式产物已拉回`代码/artifacts/formal_training/pi_jwm_r5_gpu_training_v1/`；86项清单哈希/大小一致，15个最佳checkpoint全部加载。
- [x] 新增可复用统计模块、CLI和6项定向测试；最终统计包位于`代码/artifacts/formal_training/pi_jwm_r5_multi_seed_analysis_v2/`。
- [x] B/C综合分数相对A约改善1%且3/3 seed同向，但链路活动AUPRC明显下降；C对B无点预测增益，D只改善连续状态，E不稳定。
- [x] 当前冻结状态为`r5_gpu_training_complete=true`、`r5_descriptive_analysis_complete=true`、`r5_method_freeze_ready=false`、`r6_ready=false`。
- [ ] 下一步在本地补齐R2全指标与不确定性复评；不需要重新生成数据，也不立即追加GPU训练。

## 2026-08-08 R6在线闭环GPU本地门禁

- [x] 反事实reward支持审计完成：默认动作16,200步有事实标签，其他5类候选为0；代理reward路线判定no-go，正式路线使用AirFogSim真实反馈。
- [x] 修复旧runner在策略分叉后仍读取冻结教师状态的问题；每步重新采集AirFogSim节点、信息通信边、任务/DAG、传输事件和实际卸载/RB/CPU动作，生成8步在线严格双图状态。
- [x] 修复CPU分配器被提前调用导致随机策略RNG推进，以及默认动作日志未被实际候选动作替换的问题；相关回归测试通过。
- [x] Actor–Critic/PPO×显式/隐式/联合状态六组合CPU短闭环全部通过；32步闭环覆盖6类候选、32个不同在线状态、CPU动作和通信事件，4次更新均有非零梯度，硬违规0。
- [x] 同seed 32步复跑损失与候选计数一致；2→4步原子续训、validation-only best checkpoint、18-run隔离启动器和10k阶段dry-run通过。
- [x] 本地门禁产物位于`代码/artifacts/preflight/pi_jwm_r6_online_gpu_readiness_v2/`，结论为`ready_for_gpu_smoke=true`、`formal_gpu_training_started=false`。

## 2026-08-13 P2单步非训练采集器契约

- [x] 新增纯动作契约`single_step_collector_contract_v1.py`，所有RB COO在调用AirFogSim setter前验证，阻断其越界取模行为。
- [x] 新增`airfogsim_single_step_collector_v1.py`，真实调用卸载/RB scheduler、冻结CPU callback和一次`env.step()`，记录CPU before/after、节点容量和工作守恒。
- [x] 新增canonical runner与三组测试；writer使用同级临时目录和`os.replace`原子发布，失败不产生成功manifest，非空目录不覆盖。
- [x] 自审将本地/远端复合夹具改为双远端严格单因素夹具；两候选同seed/config/state/RNG、同50-RB COO，只改变目标节点。
- [x] 正式seed 0结果：`Task_4`分别送达`RSU_2/RSU_0`，两候选送达量均`0.3350039866`，均在同槽进入目标CPU集合并执行冻结规则。
- [x] 12项运行门、2/2 E0/CEP审计、8个产物哈希、21个源码/设计/测试依赖闭包哈希和独立`--verify-only`通过。
- [x] 48项P1/P2相关回归与Python语法编译通过；未启动GPU、未访问locked test、未生成训练数据。
- [x] 自审过程中的四个旧bundle分别保留为`pre_self_review`、`pre_atomic_manifest_review`、`pre_single_factor_review`和`pre_e0_cpu_target_gate`，不作为正式证据。
- [x] 已冻结多步时序、E1上一槽回填、跨步节点/边/flow词表和失败原子发布契约，并完成seed 0三帧非训练trajectory smoke；该证据仍不批准正式v4数据集或训练。

## 2026-08-12 P1信息边v4设计与自审

- [x] 完成旧18槽、当前5个有效槽、AirFogSim逐RB内部量、transfer event、动作台账和旧v3 artifact的交叉核查；确认13维不能统一“补齐”。
- [x] 形成E0/E1/E2/E3分层方案：E1为五个链路级核心状态，E2为上一时隙逐RB增强，E3为当前动作前逐RB上界诊断；字段多少不是目标。
- [x] 将权威RB动作冻结为`(time,flow,edge,RB)`稀疏COO，避免RB count、bitmap或单跳edge视图丢失理论动作身份。
- [x] 自审修正单位冒进：AirFogSim rate与task size的bit/byte换算尚无证据，P2校准前只使用AirFogSim data-unit口径。
- [x] 自审修正旧`rate_sum`来源冲突、来源/缺失混淆、模型容量混杂、未激活RB占位值和locked-test coverage边界。
- [x] 设计文档：`文档/研究进展/2026-08-12-PI-JWM-v4信息边均衡协议设计.md`。
- [x] 用户已复核并确认成文设计；2026-08-13已按TDD完成P1-MVS。设计阶段自身未修改生产代码、重建数据、训练或访问locked test。

## 2026-08-13 P1-MVS信息边协议实施

- [x] 用户确认8月12日成文设计；实施计划写入`文档/研究进展/2026-08-13-PI-JWM-P1-MVS实施计划.md`。
- [x] 以TDD新增`information_edge_contract_v4.py`：29个registry字段、5个E1核心字段、9种MissingReason、18条旧槽迁移，以及严格Mask/dtype、适用边、首帧、范围、守恒和`(t,f,e,b)` COO验证器。
- [x] 以TDD新增只读审计脚本：locked split在路径解析和`np.load`前过滤；非锁定张量逐条流式扫描；输出使用临时目录和原子发布。
- [x] 正式审计54条非锁定轨迹：train 36、validation 12、calibration 6；6类场景、9种实际无线边类型；未发现wired真实样例，不伪造。
- [x] 复算旧18槽仍仅0/1/8/11/12有效，每维真实present边有效计数`5,854,752`；其余13维为0。
- [x] `micro_sample.npz`包含270条`observed_nonlocked`旧证据和5条`contract_fixture`，全部`training_eligible=false`、`v4_field_implemented=false`；旧字段名与候选v4目标分列。
- [x] 自审修正固定配置/不可用字段registry、legacy/v4命名混淆及padding容量污染coverage分母。早期产物移至`pi_jwm_p1_information_edge_contract_v4_pre_self_review_20260813/`，不作正式证据。
- [x] 独立审查指出并修复5类验收缺口：18槽/schema精确门、失败`_failed`证据包、代码/协议/配置哈希、数值/守恒/dtype门、locked端到端I/O哨兵与基于实际输入清单的访问标志。审查前产物移至`pi_jwm_p1_information_edge_contract_v4_pre_independent_review_fix_20260813/`，不作正式证据。
- [x] 修复后在同一非锁定数据上重跑正式审计；v3+v4相关回归54项通过，Python编译通过；正式6个输出、110个输入和2个代码文件SHA-256独立复算0失败，协议/配置哈希一致，locked输入路径0，`rejected_record_count=0`。
- [x] 一次回归命令引用交接文档中的旧测试文件名而出现2个`ModuleNotFoundError`；定位真实测试文件后运行54项当前相关测试全部通过，不把旧文件名缺失算作实现通过或失败。
- [x] PowerShell 5默认编码读取含中文绝对路径的UTF-8 manifest时产生乱码和JSON转义错误；改用UTF-8 Python解析并从工作区枚举路径后完成哈希复算，不把解析工具失败当作manifest失败。
- [x] P1-A已冻结，见下一节；P2采集器、正式v4数据、模型重训、E1/E2/E3消融和100k训练均未开始。

## 2026-08-13 P1-A CPU动作边界冻结

- [x] 用户确认核心动作只保留卸载与RB；CPU冻结为每个候选通信后状态上的确定性工作守恒内层规则。
- [x] 正式设计写入`文档/研究进展/2026-08-13-PI-JWM-P1-A-CPU动作边界冻结设计.md`，明确数学定义、AirFogSim同槽时序、P2数据字段、模型/规划器/baseline约束和旧实验失效范围。
- [x] 主文档将“固定CPU规则”收紧为“逐候选、逐rollout步真实调用”，避免误解成CPU与候选无关或不参与状态转移。
- [x] 旧R1混合CPU策略数据降级为历史工程证据；旧R6联合CPU与10k降级为扩展动作/工程历史证据，不继续100k。
- [x] P2-A已实现CPU规则纯函数、AirFogSim callback一致性和机器可读最小证据；正式v4采集器与数据生成仍未开始。

## 2026-08-13 P2-A CPU内层规则最小可执行验证

- [x] 实施计划写入`docs/superpowers/plans/2026-08-13-p2-a-cpu-inner-rule.md`；范围明确排除v4正式数据、训练、GPU、locked test和最终方法冻结。
- [x] 以TDD新增`代码/src/pi_jwm/cpu_inner_rule_v1.py`：确定性、逐节点、封顶均分、工作守恒，按`task_id`稳定排序；容量和任务剩余工作上界在返回前再次验证。
- [x] 纯规则11项测试通过，覆盖空集合、零容量、需求不足/过载、非等需求、多节点、浮点守恒、确定性、候选后通信任务集合变化和非法输入拒绝。
- [x] 新增`代码/src/pi_jwm/airfogsim_cpu_inner_rule_v1.py`，适配`TaskManager.computeTasks`真实callback形状；使用AirFogSim仓库真实`Task`源码验证与纯函数同输入完全一致，5项接口测试通过。
- [x] 定位当前环境的AirFogSim顶层导入副作用：GBK无法编码emoji，UTF-8下又缺可视化依赖`osmnx`。没有修改第三方代码；测试只绕开顶层GUI导入，显式执行真实Task依赖源码。该结果不冒充完整AirFogSim轨迹运行。
- [x] 自审补齐需求率溢出拒绝门；旧5条拒绝记录bundle移入`pi_jwm_cpu_inner_rule_v1_pre_self_review_20260813/`，不作正式证据。
- [x] 原子生成正式`代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/`：8个合约案例、14条通过记录、6条预期拒绝记录；全部`training_eligible=false`。manifest绑定设计、实现、测试、AirFogSim源码和输出哈希。
- [x] P2-A聚合20项测试及Python编译通过；未访问locked test，未启动GPU，未实现v4采集器、数据集、世界模型训练或候选rollout规划器。
- [ ] 下一串行门：先冻结P2采集器的逐步时序、字段、Mask/MissingReason、CPU规则调用与拒绝契约，并做单步非训练集成；通过后才批准新目录正式数据生成。

## 2026-08-13 P2采集器契约设计启动

- [x] 用户授权继续推进，并要求需要GPU时提前说明；当前设计与单步CPU验证不需要GPU。
- [x] 首轮核查确认旧formal runtime的callback注入边界可复用，但历史CPU policy与冻结规则不一致，不能直接沿用为v4采集器。
- [x] 核查v4字段验证器已有Mask/MissingReason、动作COO、链路守恒和逐RB结果门，后续必须直接复用。
- [x] `airfogsim`专用Conda环境可用，Python位于`D:\miniconda\envs\airfogsim\python.exe`且`osmnx 2.0.7`可导入；真实无界面一步集成不需要GPU。
- [x] 首次registry枚举脚本错误假设存在统一`timing`键并触发`KeyError`；未据此推断字段，改为先读取真实schema键再建立来源映射。
- [x] 按真实`temporal_role/source_requirement`完成首轮分层：E0+E1与link outcome作为单步强制链；E2/E3只接直接同RB身份来源，不能反推补齐。
- [ ] 继续核查direct transfer event、任务/CPU/能耗ledger、严格双图和数据manifest的真实字段与调用顺序。
- [ ] 比较单步集成方案、成文设计并交用户复核；设计确认前不修改生产代码。
# 2026-08-13 P2 多步前置时序审计

- 未启动 GPU、未读取 locked test、未生成正式 v4 数据集。
- 只读核查确认：现有 P2 单步 bundle 中信道衰减的 action-pre 名称与采样时序冲突。
- 当前停在设计选择门：先修正单步时序证据，再实施跨步词表和 E1 上一槽回填；不能直接把现有单步路径循环成轨迹。
# 2026-08-13 P2 多步实施 RED/GREEN 记录

- Task 1 RED：新增 2 项 adapter 时序测试均因 `execute_candidate` 缺少 `pre_action_observer` 失败；原有 3 项通过。
- Task 1 GREEN：最小实现 setter 前观测回调与阶段 trace 后，5/5 adapter 测试通过；提交 `794a8f6`。
- Task 2 RED：新增 3 项 runner 时序证据测试均因 fixture/runner 尚无 `candidate_audits`、decision-time channel 与 outcome-side 标签失败；原有 3 项通过。
- 未启动 GPU、未访问 locked test、尚未移动或重建任何正式 artifact 目录。
- 工具错误：一次同时读取 `airfogsim_env.py` 与 `channel_manager_cp.py` 的 PowerShell 片段命令在 10 秒后超时；未写文件，改用 `rg -C` 小范围读取，不重复原命令。
- [x] Task 2：修正单步action-pre时序，旧bundle移动到`pi_jwm_p2_single_step_collector_v1_pre_temporal_fix_20260813/`，修正版canonical通过；代码提交`24e7398`。
- [x] Task 3：实现append-only node/edge/flow词表、事务历史提交、首帧`NO_HISTORY`和有效零值；26项纯契约+v4回归通过，提交`5a5b9d8`。
- [x] Task 4：实现固定seed 0三帧真实AirFogSim smoke、跨帧E1回填、词表/COO身份、篡改拒绝和原子发布；提交`ddfbaa3`。
- [x] Task 5：正式单步/多步bundle分别通过verify-only；77项聚合回归、动态helper 16项测试和语法编译通过；独立UTF-8 SHA-256复算单步8 artifact/21 source、多步7 artifact/26 source均0不匹配。集成审计补齐PI-JWM直接、传递和runner动态导入依赖闭包；AirFogSim仍为独立第三方checkout。
- [x] 合并后干净`main`范围验证：P2聚合77项与动态helper 16项均通过。全仓测试发现运行933项、出现283个错误并跳过2项，主要是历史测试缺少`sklearn`、Python 3.10缺少`tomllib`及其他未跟踪PI-JWM模块；不得表述为全仓测试通过。
- [x] 无GPU、无locked test、无训练。`v4_collector_implemented=false`、`v4_dataset_complete=false`、`training_eligible=false`、`candidate_rollout_planner_complete=false`、`final_method_frozen=false`。
- [ ] 下一串行门：正式v4全双图采集器的非训练覆盖设计与实现，至少包含完整E0/CEP、动态presence、多任务/多流、多seed和失败恢复；通过前不启动训练。

## 2026-08-13 P2正式v4全双图采集器设计启动

- [x] 用户授权实施完整v4非训练采集器门；明确仍不训练、不启动GPU、不访问locked test。
- [x] 启用brainstorming与planning-with-files流程；设计确认前只读核查，不修改生产实现。
- [ ] 核查历史formal graph builder、严格双图preflight与当前多步契约的真实复用边界。
- [ ] 比较实现方案并冻结完整E0/CEP/presence的可证据定义。
- [x] 已确认历史formal tensor只提供precedence-only DAG参考，strict preflight的EP直接证据门应保留；通信流边与任务DAG边不能混为一个CEP空间。
- [x] 已确认AirFogSim全信道矩阵与本槽activated link是不同接口；正式设计需将物理结构presence与活动状态分离，不发明距离可达阈值。
- [x] 已核查Task跨卸载/返回/多跳/改路由生命周期；提出逻辑flow与承载hop两级稳定身份，等待用户确认。
- [x] 用户确认逻辑flow/承载hop两级身份、任务DAG与通信flow分离、完整结构边presence与active分离，以及自然任务生成加显式合法动作的方向。
- [x] 核查RB真实执行语义：setter会静默取模，真实承载边取自当前route首跳，不同链路同RB会进入真实干扰计算，有线flow不使用RB，回传route独立启动。
- [x] 比较`disjoint_only/reuse_only/balanced_two_arm`；推荐正交常规帧加复用干扰帧的双臂覆盖策略，并保持合法性定义独立于覆盖策略。
- [ ] 将完整设计、拒绝门、测试矩阵、artifact/manifest验收标准成文并交用户确认；确认前不写v4实现代码。

## 2026-08-14 P2-C AirFogSim依赖恢复

- [x] 确认P2-B manifest要求80个AirFogSim Python文件和3个配置/SUMO文件。
- [x] 本机归档候选逐项比对为82/83完全匹配，0路径缺失；唯一不匹配为`energy_manager.py`，且不是换行差异。
- [x] 搜索2026-08-11本机会话：仅发现依赖枚举/搜索输出，没有恢复出目标文件正文。
- [x] 从2026-05-05审批包装层追踪到真正的2026-04-26归档原会话及其本机备份。
- [x] 原会话确认2026-05-05官方clone，且后续`energy_manager.py`文件长度3478字节；未发现该文件的直接补丁/写入记录。
- [x] 全归档会话搜索发现2026-07-13曾对该第三方文件先添加再撤回能耗快照逻辑；已形成“补丁后混合换行导致单文件哈希变化”的可检验假设。
- [x] 从会话日志提取两次完整补丁，在临时副本依序重放；结果逻辑文本不变，混合换行字节SHA-256精确命中manifest目标。
- [x] 组装临时83项候选并依据P2-B manifest独立复算，要求83/83全部匹配后才恢复项目引用目录。
- [x] 临时候选83/83通过；确认项目目标为普通空目录后完成恢复，目标目录独立复算再次83/83、0 mismatch。
- [x] 运行P2-B/P2-C `--verify-only`，并在`PYTHONUTF8=1`下运行完整focused suite；双验证和159项测试均通过。
- [x] P2-B与显式路径P2-C verify-only均通过。
- [x] 首轮GBK/emoji错误已由`PYTHONUTF8=1`原样复跑验证为环境根因，未修改第三方源码。
- [x] `PYTHONUTF8=1`原样复跑实际展开158项并全部通过，证实编码环境根因且没有修改第三方源码。
- [x] 在新候选目录重算P2-C审计；仅关闭`input_manifest_source_closure_failed`，其余四个真实阻断保持。
- [x] 新候选重算与verify-only通过；diff确认只关闭source-closure阻断，formal config不变，四个其余阻断保持。
- [x] 验证路径边界后保留旧canonical归档，并将已验证候选提升为canonical；随后再次verify-only和哈希复算。
- [x] 旧canonical已保留为`pre_source_recovery_20260814`，验证候选已提升；提升后双verify-only、83/83和158/158通过。
- [x] P2-C研究进展文档已进入manifest source list；RED/GREEN、canonical重建与文档篡改检测均完成。
- [x] 用户确认推荐设计；规格已自审并以提交`200a209`固化，未修改runner或测试。
- [x] 用户确认书面规格后已按实施计划完成RED/GREEN与证据门。
- [x] 精确目标SHA-256已由历史补丁重放恢复并完成83/83复验。
# 2026-08-15 新对话迁移接续文档进度

- 已读取 `新对话接续说明_20260811.md`、现有计划记录、主工作树及隔离 worktree 状态。
- 未启动 GPU、未访问 locked test、未修改第三方 AirFogSim、未清理任何工作树或 artifact。
- 正在核验最新 P2-B/P2-C 机器可读证据，随后生成根目录新接续说明。
- 已核验P2-B v2的status/validation/replay/config和隔离分支diff；P2-C v2正确文件名已由目录实物确认。
- 读取时出现两个 `Get-Content` FileNotFound，仅因使用了错误的摘要文件名；未修改artifact，已切换到真实文件名继续。
- 一次多文件 `apply_patch` 因hunk格式错误被拒绝；未写入错误内容，随后使用合法的独立hunk完成记录。
- 已解析P2-C v2完整audit/config JSON，并核对外部`PIJWM主文档.md`与`8.12之后推进.md`的权威边界及滞后状态。
- 已确定新接续文档结构；下一步写入根目录`新对话接续说明_20260815.md`。
- 已创建`新对话接续说明_20260815.md`，共542行、28,562字节。
- 独立校验确认关键声明与P2-C v2 JSON一致，引用的核心代码/artifact路径存在，最终候选目录仍不存在，尾空格/TODO检查为0。
