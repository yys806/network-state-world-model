# PI-JWM v11 Selector Findings

## 2026-08-15 仓库顶层目录重构设计

- 用户要求：`文档/知识库`迁到根目录并改名`记录/`；后续项目更新统一维护此处；根目录散落Markdown归类；`代码/`改名`code/`；原`文档/`拆成根级`paper/`、`literature/`、`meeting/`，另建根级`docs/`承接杂项；测试和临时材料归入`code/`或`docs/`。
- 该变更会同时影响命令路径、Python路径、测试发现、文档链接、Git跟踪、忽略规则、机器artifact source manifest和后续交接入口，不能只做文件移动。
- `brainstorming`硬门要求先探索现状、逐项澄清、比较方案、形成并经用户确认的设计，再写实施计划；确认前不实施迁移。
- 根目录当前有9个Markdown：`AGENTS.md`、`README.md`、`本地计划表.md`、三个`新对话接续说明_*.md`以及`task_plan.md/findings.md/progress.md`。其中AGENTS必须留根才能自动生效，README留根符合标准项目入口；其余7个均可作为`记录/`候选。
- 原`文档/`顶层只有`文献/`、`研究进展/`、`知识库/`、`组会/`和导航README；初步职责映射分别对应新`literature/`、待判定的`paper/或docs/`、`记录/`、`meeting/`以及新根README导航。
- 原`代码/`顶层已是`src/`、`scripts/`、`tests/`、`reference/`、`artifacts/`和两个requirements文件，改名为`code/`本身不需要内部再拆结构，但会影响大量路径引用。
- 根目录还有`.ruff_cache/`和`.worktrees/`；前者是可清理缓存，后者是Git隔离工作树行政目录，不能当普通tmp移动。
- `文档/文献`有89文件、约313.18 MB（78 PDF）；可整体映射到根`literature/`。`文档/组会`有1365文件、约142.00 MB，可整体映射到根`meeting/`。
- `文档/知识库`仅3个Markdown（主文档、8.12之后推进、README），可整体映射到根`记录/`；根上的本地计划、三份接续说明及三份planning日志也适合并入`记录/`的子目录。
- `文档/研究进展`有91文件、约6.28 MB：顶层约50份设计/实施/结果Markdown和历史研究进展TeX/PDF，另有`归档/`与`figs/`。它混合“研究论文材料”“设计与计划”“历史项目说明”，不能不加筛选地全部叫`paper/`；应按研究稿件/过程记录/杂项归档拆到`paper/`、`记录/`、`docs/`。
- 主工作树的正式测试已经位于`代码/tests/`，无需另迁；`代码/artifacts/tmp/`属于生成临时产物，应留在未来`code/artifacts/tmp/`或清理，而不是放到文档目录。大量`__pycache__`和根`.ruff_cache`是缓存，可清理而非归档。
- `.worktrees/`下的tests/cache属于其他Git工作树的独立内容，不能随主工作树目录迁移；需先审计三个worktree是否仍活动，再决定保留或单独收尾。
- Git当前跟踪前缀：`代码`561项、`文档`94项、旧`docs`19项（当前为删除状态）、根Markdown 9项。目录重构将是大规模Git rename，必须用分阶段提交和路径引用门验证。
- `pyproject.toml`把setuptools包根固定为`代码/src`；`.gitignore`把artifact、AirFogSim、文献、组会和研究进展二进制规则全部绑定旧顶层路径。改名必须同步这两个配置，否则安装、忽略边界和Git体积保护都会失效。
- tracked文本中有95个文件引用旧`代码`路径、29个文件引用旧`文档`路径；其中大部分在历史研究记录，但至少26个代码侧文件和多项测试/脚本存在活跃硬编码。
- 若干测试把`代码/...`不仅当路径，还当manifest source key的协议字符串进行断言；旧canonical artifacts也绑定这些字符串。不能机械全局替换历史manifest，否则会改变已冻结证据；推荐对“现行路径”和“历史证据路径标识”分层处理。
- 现有三个非主worktree仍注册并指向`.worktrees/`下旧提交；它们是独立Git工作树，主工作树改名不会自动重写这些分支内容。迁移设计需保留其行政目录，且不得把其中tests/tmp当主项目散落文件移动。
- 根`docs/`目前没有活动物理目录，但Git记录19个`docs/superpowers`文件为既有删除状态；新`docs/`可复用这个顶层名称，但不能误恢复已删除的superpowers材料。
- 用户确认推荐边界：`paper/`只放正式论文稿件、LaTeX、投稿材料和论文配图；现有研究设计、实施计划与结果报告进入`记录/研究进展/`，模板和杂项进入`docs/`。
- 用户确认具体归类与回滚规则：正式论文材料单独进入`paper/`，过程记录进入`记录/`，文献/组会整体迁移，旧manifest不改字节，分阶段失败可按映射反向恢复。
- 2026-08-16迁移前新鲜基线：P2-B v1、P2-B v2 candidate、P2-C v1、P2-C v2 pre-document均退出1；前三项直接由已删除`docs/superpowers`source触发，v2 pre-document还存在published audit与fresh recomputation差异。目录重构验收只能要求`layout_induced=0`，不能宣称四门通过或恢复已删除材料。
- 用户选择第3种分阶段原子迁移方案：新目录物理落地、现行路径更新、历史manifest字节不改，通过兼容解析和分阶段验收保持可追溯性。
- 用户确认迁移顺序与兼容策略：先文档归类、后`代码`改名、再更新路径/验证器、最后清理缓存；不保留长期旧目录别名，历史manifest只通过兼容解析验收。
- 用户已确认2026-08-16验收修订：四个证据门的既有失败继续如实保留，迁移只以不新增目录布局导致的错误（`layout_induced=0`）为合格条件。
- 迁移实测映射共16,846条：code 15,290、literature 89、meeting 1,365、records 81、docs 18、paper 3。迁移后逐项大小与SHA-256校验无缺失、残留源或差异。
- 全量测试首次发现4个归档审计失败和2个CPU预检错误仍由现行测试写死旧`文档/`路径导致；改为`记录/`、`paper/`和`docs/`真实位置后定向测试通过。随后又发现P2-C测试用旧`D:\shen\网络组\代码`常量造成真实bundle测试被跳过，改为当前`code/artifacts/`后5项核心与4项runner测试全部执行并通过。
- 最终全量测试运行1,360项，无断言失败；7个错误均在导入`code/reference/AirFogSim/`时缺少`traci`。该环境事实与目录迁移错误分开记录，不能宣称全仓全绿。
- 文献库独立复算确认78个PDF全部以`%PDF-`开头，索引路径、索引SHA-256与文件一致，78个哈希唯一，checksum 78行；没有`待整理`目录。
- `meeting/`含Office锁文件和LaTeX生成物等历史字节证据，数量统计会受默认隐藏/忽略规则影响；验收以映射中的1,365个目标逐项存在和哈希匹配为准，不以单一`rg --files`计数替代。
- `code/artifacts/tmp/gpu_upload_eff9385`包含完整旧仓库副本、补丁和压缩包，虽位于tmp但不是可随意删除的纯缓存；保留在规范位置以保护来源链。RRM仅做存在性确认，未读取、移动或改写其内容。

## 2026-08-15 手动下载论文入库与Zotero遗留清理

- 用户已把此前列出的14篇手动下载论文放入`文档/文献/待整理/`，要求完成分类入库并清除不再需要的Zotero相关文件。
- 本轮以`文档/文献/README.md`与`文档/文献/文献索引.csv`为权威；每篇PDF仅保留一个主分类路径，接受前必须核验`%PDF-`签名并去重。
- 当前工作树已有`docs/superpowers/`下多项tracked deletion；这些不是本轮产生，必须保持隔离。
- `待整理/`当前恰有14个PDF，与手动下载清单数量一致；其中`2107.07511v6.pdf`需用正文/DOI确认是否为Conformal Prediction，其余文件名可初步对应13个缺失题名，但仍不能仅凭文件名入库。
- 当前README仍记录64个PDF、14篇待下载，并把`PIJWM文献库.bib`、两个`zotero_pijwm_*.json`、`zotero附件清单.csv`等迁移快照作为保留项；这些状态需要在本轮归档后重建。
- 明确的Zotero遗留候选至少包括`PIJWM文献库.bib`、`zotero_pijwm_collections_20260815.json`、`zotero_pijwm_export_20260815.json`和`zotero附件清单.csv`；`初始PDF迁移清单.csv`、`待补全PDF候选.csv`、`自动下载结果.csv`及入库审计也可能仅服务一次性迁移，必须检查当前索引是否依赖后再清理。
- `PIJWM文献管理说明.md`规定入库顺序为DOI/arXiv/规范化题名去重、`%PDF-`签名、首页题名、唯一主分类、SHA-256去重，然后同步CSV/Markdown/哈希/状态/缺失清单；本轮严格按此执行。
- 仓库内没有可复用的文献索引重建脚本；现行规则只在`AGENTS.md`、文献管理说明和文档导航中声明，因此需基于既有文件真实schema做最小更新，并独立复算验收。
- 管理说明第6节把四个显式Zotero快照以及`自动下载结果.csv`、`初始PDF迁移清单.csv`列为一次性退役快照。用户本轮已明确要求清除Zotero相关文件，因此在确认当前本地索引不依赖后可将这些快照送入回收站，并同步移除过时导航。
- 14个待整理文件全部以`%PDF-`开头，均可由pypdf 6.0.0解析，共6—51页不等；13篇首页题名和正文DOI与目标记录逐项一致。
- `2107.07511v6.pdf`首页为Angelopoulos与Bates的`A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification`，即目标Conformal Prediction工作的arXiv版本；文件内无正式DOI，不能表述为`10.1561/2200000101`出版社版，入库来源必须显式标记为用户提供的arXiv版本。
- 其余13篇均从正文抽取到目标DOI：JSAC 2篇、TNSM、TVT、JIOT 3篇、TCCN、TWC、TMC 2篇、TPDS、ICC Workshops各一篇；PDF元数据/首页题名相符。
- 14个新PDF的SHA-256彼此唯一，且与现有64个PDF零重合；全部可作为新增主阅读副本，无需删除重复PDF。
- 现有索引只有`available`与`manual_required`两类状态；本轮正式版13篇可用`available/manual_user_download`，Conformal的arXiv副本也可作为`available`，但Source需用`manual_user_arxiv_version`明确版本边界。
- 旧`文献索引.md`和`需要手动下载.md`仍链接BibTeX/Zotero快照并包含历史Zotero key建议；归档完成后前者需改为78篇全覆盖、本地入口，后者应改为空缺状态且移除Zotero指引。
- 14个目标文件经独立路径复核均位于文献根目录内、父目录存在、文件存在；移动后`待整理`为0。
- 当前分类PDF总数为78，SHA-256唯一值也是78，合计313,064,419字节；不存在新增重复或遗失。
- 新分类统计：01=5、02=20、03=15、04=13、05=11、06=5、99=9；对应大小分别为8.96、57.75、86.72、51.14、58.58、12.14、23.27 MiB。
- 八个一次性迁移/Zotero候选文件合计369,240字节：BibTeX、两个Zotero JSON、附件清单、初始迁移清单、待补全候选、自动下载结果和旧入库审计。仓库现行代码没有运行时引用；已发现的外部引用仅是README/管理说明/旧计划和本轮记录，均属于导航或历史陈述，可同步更新而不影响PDF或索引。
- 当前CSV已更新为78条`available`、0条`manual_required`；14条新增来源为13条`manual_user_download`和1条`manual_user_arxiv_version`，逐项本地路径、字节和哈希校验0错误。
- 八个一次性迁移/Zotero文件已按精确绝对路径送入Windows回收站，活动路径剩余0；操作可恢复且未清空回收站。
- 全仓文件名复查发现另有历史目录`代码/artifacts/literature/zotero_import_20260722/`，包含旧导入PDF与下载账本；它不属于现行`文档/文献/`，需先做哈希覆盖和引用核对再清理。
- `本地计划表.md`顶部仍陈述64篇/14篇及快照保留，旧2026-07-24段仍链接已清理的入库审计；这些是过时计划陈述，需同步为78篇/0缺失并注明旧审计已由现行索引/验收替代。
- 历史`代码/artifacts/literature/zotero_import_20260722/`含16个文件、14个PDF、66,693,588字节；其中10个PDF与现行本地库逐字节同哈希，4个是Polese/Wu相关论文的不同字节版本，必须先按题名/DOI确认现行库已有主阅读版本后才能删除整个目录。
- 该历史目录无现行代码引用；唯一外部命中是一个旧`source_inventory.json`，其中记录的路径本来就是已迁移前的`D:\shen\网络组\...`绝对路径，因此当前目录不是该历史artifact可执行依赖。删除后仍需保留其只读清单作为历史事实，不改写旧artifact。
- 4个不同字节旧PDF的首页题名分别与现行索引中的Understanding O-RAN、Colosseum和两篇Blockage论文一致；现行主副本绑定正式DOI，旧目录副本无正文DOI，因此没有独有论文身份需要保留。
- `代码/artifacts/literature/zotero_import_20260722/`已整体送入Windows回收站：活动路径移除16文件、66,693,588字节；`source_inventory.json`未修改，RRM未触碰。
- 独立验收重新遍历七类目录并逐项复算：78条索引、78个available、0个manual、78个PDF、78个唯一哈希、313,064,419字节、78条checksum、78个唯一索引路径，签名/路径/字节/哈希错误均为0。
- 验收同时确认`待整理`PDF为0、最大绝对路径长度157、8个迁移文件剩余0、全仓Zotero命名文件/目录剩余0；Conformal arXiv版本边界已写入状态与验收JSON。

## 2026-08-15 本地文献库迁移基线

- 官方OpenAI文档说明Codex从用户级技能位置加载全局技能；本机唯一名为`using-superpowers`的用户级目录是`C:\Users\Lenovo\.codex\skills\using-superpowers`，`C:\Users\Lenovo\.agents\skills`和仓库内均无同名副本。
- `文档/文献/`已有若干散放PDF、`本地论文/`、旧研究资料目录和四份文献治理Markdown，整理时必须保留并按内容去重，不能只导出Zotero附件。
- Zotero本地数据目录为`D:\禹尧珅\人工智能知识库\科研`；只通过Zotero API/MCP读取和写入，绝不直接修改`zotero.sqlite`。
- 本轮采用七类目录，与PIJWM Zotero七个既有子集合一一对应；PDF选择唯一主分类，多集合成员关系写入索引。
- Zotero个人库本地ID为`0`，共205项；PIJWM根集合`MZ9JQ2I6`及七个子集合键与`AGENTS.md`记录完全一致。RRM根集合`8J8EY24G`独立存在，本轮不读取、不迁移、不删除。
- PIJWM根集合共有56个顶层文献条目；Zotero覆盖审计显示30项带PDF、26项缺PDF，当前覆盖率53.6%。缺失项中18项已有DOI，8项需按题名/arXiv等进一步解析。
- `文档/文献/`当前有42个文件，其中30个PDF、约99.3 MB；30个PDF的SHA-256均唯一，14个PDF散放在目录顶层，其余位于历史子目录。整理时需要和Zotero的30项附件按哈希/题名交叉去重。
- Zotero的32个PDF附件记录包括2个`linked_url`和30个导入附件；按标准`storage/<attachment-key>/<filename>`路径核对时，24个真实文件存在、6个记录对应文件缺失（DeepSense 6G、latent wireless dynamics、JEPA、TD-MPC2、Dreamer、World Models），不能仅凭Zotero的PDF图标认定本地可读。
- Zotero附件路径工具对上述6项逐项确认均为`missing on disk`。其中Dreamer与World Models在现有本地历史目录已有候选PDF，需用题名和PDF哈希/元数据匹配；其余4项进入自动补全候选。
- 多个PIJWM条目同时属于导师或其他非PIJWM集合；删除PIJWM集合树不会级联删除条目，必须保留这些跨集合记录，不能批量删除item。
- Zotero本地API可直接导出56条完整BibTeX（约55 KB）和90条PIJWM相关JSON记录（56顶层条目、32个PDF附件记录、2个note），足以在删除集合前保留结构化快照。
- Zotero磁盘上实际存在的24个附件文件与现有本地30个PDF比较后有6个SHA-256完全相同：Ding综述、低空网络综述、UAV群ISCCC、RoboScape、dependency-aware卸载、DT到world model。整理时复用本地副本，不重复复制。
- `paper-fetch`本机版本0.14.1、schema 1.10.1可用；后续显式设置`PAPER_FETCH_NO_SCIHUB=1`并保持public模式。
- 首轮28 DOI批处理的根因不是下载链整体失败，而是Windows默认`sys.stdout.encoding=gbk`无法输出Unicode `ğ`；`PYTHONUTF8=1`下输出编码变为UTF-8。NDJSON证明14项`download_ok`、13项`not_found`，仅`10.1109/JIOT.2020.3030926`在`source_hit`后缺终态，需要单项UTF-8重试。
- 单项UTF-8重试确认`10.1109/JIOT.2020.3030926`的PolyU来源当前返回HTTP 502，因此归为手动下载而非编码故障。4个无DOI题名均高置信解析到arXiv并成功下载：Deep Ensembles、Graph World Model、GNS和PlaNet。
- 自动公开来源最终新增18篇PDF；结合迁移前已确认的24篇本地可读Zotero条目，当前56项中42项已有本地PDF，剩余14项需要手动下载。
- 七类目录现有64个PDF且64个SHA-256全部唯一：01类4篇、02类17篇、03类12篇、04类8篇、05类10篇、06类4篇、99类9篇。除七类目录外不再散放PDF。
- 14篇手动项包括13篇public OA未命中和1篇PolyU来源HTTP 502；以IEEE论文为主，清单保留Zotero key、DOI、原始URL和目标分类。
- 本地退役前验收`passed=true/errors=[]`：64个PDF、64个唯一SHA-256、78条索引、14条手动清单、90条Zotero JSON记录、8条集合记录、56条BibTeX和64条checksum全部一致；最长PDF路径157字符。
- Zotero云端PIJWM根集合删除成功，七个子集合随根集合删除；云端与本地回读均确认8个旧key为0。示例条目`HEMB6GN5`仍存在且保留导师集合归属，RRM集合`8J8EY24G`在云端与本地均保留。
- `.gitignore`继续排除整个`文档/文献/`，因此64个PDF、原始Zotero快照、CSV索引和本地README均只保存在当前机器，不会进入公开GitHub；仓库只提交治理入口对本地权威目录的新说明。

## 2026-08-15 Repository governance baseline

- 当前主工作树位于`D:\shen\PKU\PIJWM`，`main`相对`origin/main`领先167个提交；治理设计和实施计划随后各形成一个独立提交。
- tracked文件473个，未忽略untracked文件222个；大量当前PI-JWM源码、测试和8月研究文档尚未纳入Git，发布前必须按所有权、测试和敏感信息逐组审计。
- 三个复制worktree的`.git`指针和Git common metadata仍引用`D:/shen/网络组`；P2 ledger的AirFogSim junction也引用旧根。这是迁移路径闭包问题，不是现有P2 artifact损坏证据。
- 根`README.md`仍停留在2026-07-11口径，包含旧根路径、已删除目录、v11 selector主线和旧60-seed训练示例，不能继续作为当前入口。
- `.gitignore`已覆盖worktree、缓存、生成artifact、AirFogSim、文献和组会目录，但需要统一格式并复核例外规则。
- 当前31个tracked deletion主要来自旧`文档/项目说明`、旧治理文档和旧论文/研究进展生成文件；在引用与替代关系审计完成前不提交这些删除，也不擅自恢复。
- `git worktree repair`已把三个worktree的Git行政路径和复制目录`.git`指针更新为`D:/shen/PKU/PIJWM`，分支与HEAD保持原值，且不再出现migration-related `prunable`。
- P2 ledger的AirFogSim junction已仅替换链接条目，现指向`D:\shen\PKU\PIJWM\代码\reference\AirFogSim`；主工作树和ledger worktree中的`examples/config.yaml`均可解析。
- 路径修复后，P2-B v2官方`--verify-only`返回`passed=true/errors=[]`，P2-C v2预文档候选官方`--verify-only`同样返回`passed=true/errors=[]`。这验证此前失败确为迁移路径闭包问题。

## 2026-08-15 Repository publish classification

- 新增顶层候选测试文件共69个；显式候选列表运行288项测试并全部通过，且运行记录为`locked_test_accessed=false`。
- 当前框架、脚本与测试已按所有权提交为`ac7d10b`；缓存发布集155个文件、约1.73 MiB，凭据模式扫描0命中，单文件均远低于GitHub 100 MiB限制。
- 现行研究文档和历史归档迁移已提交为`ae03b0e`；大多数旧`文档/项目说明`内容被Git识别为100%重命名，旧审计工具与主文档也保留重命名关系。三个已无引用且被现行治理取代的2026-07-11治理文件维持删除。
- 归档路径审计测试1项通过。IEEE官方模板中的既有行尾空白按原字节保留，不用格式化改写第三方历史文件。
- `airfogsim` Python 3.10环境的全量1310项发现仍有275个环境错误，主要因缺少`scikit-learn`且Python 3.10无`tomllib`；该结果不能表述为全量测试通过。最终阶段将用具备这两个依赖的`D:\miniconda\python.exe`重新区分环境问题与代码问题。

## 2026-08-15 P2 ledger closure and merge

- P2-C v2预文档闭包从ledger独立重算得到natural-reference 120/120 accepted、0 rejected、0 quarantined；5字段E1矩阵5760行，三个正式数据冻结阻断保持。
- `d6f776a`只修改ledger证据文档；新生成final candidate后，audit JSON和candidate config与预文档候选逐字节相同，manifest只有该文档source hash变化，所有状态保护旗标仍为false。
- 隔离分支 focused suite为99 tests、3 skipped；扩展到24个明确P1/P2模块后为205 tests、3 skipped，合并到main后同样为205/3。v1/v2四个官方verifier合并前后均返回`passed=true`。
- 非破坏性merge为`e5ad8e4`；当前main已包含P2 ledger/P2-C v2审计代码和文档。final candidate artifact继续保留在本地忽略目录，不进入GitHub。
- 最终编译审计通过；`D:\miniconda\python.exe`全量1350项仅7项因缺`traci`报环境error、3项skip且无断言failure；在`airfogsim`环境补跑对应4模块39项全部通过。旧`test_project_configuration.py`目录断言已由`0cc91ce`按当前归档结构修正。
- 最终tracked发布集为683个文件、14,145,456字节，最大单文件2,086,659字节；10 MiB/100 MiB文件均为0，凭据模式文件为0，现行权威文档旧根引用为0。fetch后拓扑为0个remote-only/184个local-only提交，非强制push成功，首轮远端ref与本地`2ff8485`一致。

## 2026-08-13 P2采集器契约设计核查

- 现有`formal_airfogsim_runtime_v1.py`的外部callback注入和异常后恢复机制可复用，但默认仍调用历史`CpuPolicyAllocator`，其策略可含deadline权重、随机权重和任务截断，不符合已冻结的`PIJWM-CPU-Inner-Rule-v1`，不能直接作为v4正式采集器。
- 现有`airfogsim_contract_adapter.py::capacity_safe_cpu_allocations`也只做最多3任务的一次均分，没有剩余工作封顶和容量再分配；它是历史守恒修复接口，不等于P2-A冻结规则。
- `information_edge_contract_v4.py`已提供严格bool Mask、整数MissingReason、适用边类型、字段数值范围、首帧历史缺失、`(time,flow,edge,RB)`动作COO、逐链路served/rate守恒和逐RB outage/noise下界验证。P2采集器应直接调用这些契约门，不能复制或弱化。
- P2单步集成必须保持AirFogSim实际时序：候选卸载/RB动作落地，通信更新与direct transfer event形成，随后对候选后计算队列调用CPU规则，最后采集任务/队列/能耗和下一状态。仅组装前后快照而没有真实调用顺序，不足以证明规则递推。
- 专用`airfogsim` Conda环境包含`osmnx 2.0.7`，可以运行真实无界面环境；base环境缺依赖只限制P2-A源码接口测试，不构成P2真实一步集成阻塞。
- v4 registry真实schema使用`temporal_role`而非统一`timing`。E1五项中，衰减均值/标准差来自当前动作前同一逐RB衰减快照；上一槽活动流数、有效速率和served data来自前一槽direct transfer event。首帧后三项必须为`no_history`。
- E2/E3是可选增强而非P2最小门的强制有效维：E3逐RB衰减可从动作前CSI/衰减矩阵直接取得；E2和当前逐RB结果只有在相同edge/RB身份下直接读取AirFogSim已计算的SINR、含噪干扰、rate和outage时才有效，不能从rate或衰减倒推缺失量。
- P2单步门应以E0+E1、当前动作COO、当前link outcome、CPU规则调用和下一任务状态为强制闭环；E2/E3按直接来源能力输出有效值或显式MissingReason。这符合“少而可靠”而不是机械补齐29项。

## 2026-08-13 P2-A CPU内层规则实施事实

- 新增`cpu_inner_rule_v1.py`，严格实现逐节点封顶均分：`d_m=remaining_work/slot_seconds`，分配满足`f_m=min(d_m,lambda_i)`与`sum(f_m)=min(capacity,sum(d_m))`；输出按节点和`task_id`稳定排序。
- CPU规则对空集合、零容量、需求低于/高于容量、非等需求、多节点和候选通信后任务集合变化均有测试；非法数值、重复任务、缺容量和节点关系不一致均fail-fast，不静默钳位成合法记录。
- 新增AirFogSim callback适配层，输入形状与`TaskManager.computeTasks`的`{node_id: [Task]}`一致，并读取`getFogProfile()['cpu']`、`getTaskCPU()`、`getComputedSize()`、`getAssignedTo()`和`getCurrentNodeId()`。适配输出与同输入纯函数完全一致。
- 当前默认Python环境导入`airfogsim`顶层时，先受GBK控制台emoji输出影响，改UTF-8后又缺少可视化依赖`osmnx`；为避免修改第三方源码，接口测试显式执行仓库真实`enum_const.py`、`mission.py`和`task.py`源码，但绕开顶层GUI导入副作用。此证据是`Task`源码接口级集成，不是完整AirFogSim轨迹运行。
- 正式预检bundle为`代码/artifacts/preflight/pi_jwm_cpu_inner_rule_v1/`：8个合约案例、14条通过记录、6条预期拒绝记录，全部`contract_fixture`且`training_eligible=false`；manifest绑定冻结设计、实现、测试、AirFogSim源码与4个输出哈希。修复前5条拒绝记录的bundle移入`pi_jwm_cpu_inner_rule_v1_pre_self_review_20260813/`，不作正式证据。
- P2-A可以解除“CPU规则入口未实现”这一局部阻塞，但不能解除P2/P4/P6/P7整体门：`full_airfogsim_trajectory_executed=false`、`v4_collector_implemented=false`、`v4_dataset_complete=false`、`candidate_rollout_planner_complete=false`、`final_method_frozen=false`。

## 2026-08-13 P1-MVS实施事实

### P1-A CPU动作边界冻结

- 用户已确认CPU不进入当前核心规划动作；`a_core`仅包含卸载与RB。CPU由每个候选完成通信更新后的实际计算队列、剩余工作、节点容量和时隙长度确定，必须逐候选、逐rollout步调用。
- 冻结规则为封顶均分的工作守恒分配：需求较小任务按剩余工作封顶，其余容量确定性重分配；无deadline/priority/未来结果/学习权重。它比一次均分更严格，也不等于与候选无关的常量。
- AirFogSim真实`step()`顺序为无线通信、有线通信、计算；`Task.compute()`按`allocated_cpu * simulation_interval`推进工作，支持上述通信后CPU规则时序。
- 旧R1数据的CPU策略在`equal_share/deadline_aware/feasible_exploration`间混合；CPU配对实验又证实规则会改变系统指标。因此旧数据不能当作冻结规则下新正式训练集，P2必须重建。
- 旧R6联合候选会按模板同时改变CPU，故只能保留为扩展动作历史证据；不允许继续旧100k。正式设计为`文档/研究进展/2026-08-13-PI-JWM-P1-A-CPU动作边界冻结设计.md`。

- v4机器registry已实现29项：E0结构/动作、5个E1核心、E2/E3逐RB增强、outcome监督、fixed config和明确unavailable的MCS；来源等级只允许`direct/derived/fixed_config/unavailable`。
- 对54条非锁定v3张量按真实`information_edge_present`、split、6类场景和9种实际无线边类型复算后，仍只有旧索引0/1/8/11/12有效，每个有效索引计数5,854,752；13维为0。未出现wired真实边，不能用fixture冒充数据覆盖。
- P1-MVS没有把旧观测改名成v4已实现字段：270条legacy观测保留旧字段名，并单列candidate v4 target；另有5条仅验证Mask语义的contract fixture。275条全部训练资格为false且`v4_field_implemented=false`。
- 正式manifest绑定110个输入（trajectory index、tensor contract、54组seed manifest和NPZ）与6个输出，独立复算0不匹配；输入路径不含locked seed目录。locked 6条只保留既有索引身份元数据。
- P1-MVS本身不证明v4采集器、数据或模型完成；随后P1-A冻结CPU边界，P2-A又完成CPU规则与callback接口预检，但P2采集器、正式数据、P4世界模型和P6候选rollout仍未实现；`v4_collector_implemented=false`、`v4_dataset_complete=false`、`v4_model_trained=false`。
- 独立代码审查发现首版验收门有5类缺口：未精确验证18槽顺序/seed schema、失败时不保留rejected证据、manifest缺代码哈希、数值/守恒/dtype门不完整、locked标志硬编码且缺端到端I/O哨兵。缺口均以失败测试复现后修复，审查前产物整体移入`pi_jwm_p1_information_edge_contract_v4_pre_independent_review_fix_20260813/`，不再作为正式证据。
- 修复后正式审计在同一54条非锁定轨迹上重跑，5/13结论未改变。正式manifest现绑定6个输出、110个输入和2个代码文件；UTF-8 Python解析独立复算全部SHA-256为0不匹配，协议与配置哈希一致，`locked_test_accessed=false`、`gpu_started=false`、`rejected_record_count=0`。
- 当前验证器已覆盖严格bool Mask、整数MissingReason、字段上下界、首帧`no_history`、COO稠密恢复、链路rate/served守恒、outage-rate和干扰噪声下界。它们是v4采集器未来必须调用的契约门，不表示这些v4真实字段已经采集。

## 2026-08-12 P1 信息边均衡协议设计启动

- 用户授权固定信息边方案，目标是“各个方面最均衡”，不是机械补齐18维。
- 主文档第2349行附近已明确：信息边特征数量不是优化目标，应通过“最小可靠集、可恢复增强集、新采集增强集”受控消融决定；更少但可靠且开销更低时优先精简协议。
- 当前设计评价轴暂定为：物理/业务语义明确、决策时可观测、跨场景稳定、动作敏感与预测价值、采集/计算开销、真实数据可迁移、缺失鲁棒性、与规则层/评价指标的非重复性。权重需结合用户目标确认后冻结。
- P0已证实五个当前有效槽也不等价于五个同等级可靠观测：`interface_available`为固定常量；`csi_mean`不是SINR；`rate_sum`是信道管理器返回速率和，不能直接改称实际吞吐。后续必须逐字段分级，不能简单把五维整体作为最终核心集。
- 现有18槽按代码分为8个`pre`、3个`action`和7个`outcome`。最终协议可以统一存储，但决策接口必须切开：`pre`中经审计可观测者进入当前状态，候选动作单独进入action，`outcome`只进入下一状态或监督目标，避免同刻结果泄漏。
- 18槽当前名称依次为：`pre.interface_available/csi_mean/channel_gain/path_loss/noise/historical_interference/historical_sinr/historical_rate`，`action.allocated_rb_count/tx_power/mcs`，`outcome.active_task_count/rate_sum/actual_interference/actual_sinr/outage/throughput/served_data`。
- 当前图构造函数只为`csi_mean`、`allocated_rb_count`、`active_task_count`、`rate_sum`依据源字段设置mask；`interface_available`被无条件设为1且mask为真。其余13槽全部为0且mask为假。因此`interface_available`需要改成由链路存在/可调度条件推导的合法字段，或移出连续特征而作为结构presence，不能继续把硬编码常量当作有信息量观测。
- AirFogSim `channel_manager_cp.py`在执行传输后内部持有各链路类型、各RB的`*_Interference`（含噪声）、`*_SINR`（dB）、`is_*_outage`和`*_Rate`（乘RB带宽后注释为Mbps）；`getCSI`返回带快衰落的信道衰减数组。说明若修改采集器并重新生成数据，SINR/干扰/outage/按RB速率具有潜在direct来源，但当前PI-JWM采集脚本没有导出这些量，暂不能标有效。
- 当前正式采集路径对信道侧只调用`getCSI`和`getRateByChannelType`；前者在源脚本中称channel state且用于`signal_power-noise-channel_state`估算SNR，更接近按RB信道衰减/损耗数组，不应泛称完整CSI。后者返回执行后的每RB速率，现有`rate_sum`再求和。
- `r6_online_observation.py`复用`source_physical_edge_snapshots`构建当前信息边，因此在线路径与离线路径共享相同五维限制；仅仅在线重采样没有自动补齐13维。
- 初步字段分层建议：最小集只容纳低成本、时序合法、真实系统较易获得的链路结构/历史质量/历史服务与动作量；仿真增强集容纳按链路和RB重新导出的SINR、干扰、outage；`tx_power/MCS`只有成为配置条件或正式动作后才可进入，不能为补维度而加入。
- `WirelessTransferEventRecorder`已直接记录端点、RB索引、逐RB速率、计划容量、传输前剩余量和`delivered_data`；`delivered_data=min(planned_capacity, remaining_before)`。因此`served_data`可由direct事件按“链路×时隙”聚合为derived/direct-runtime outcome，当前缺失是teacher graph未做该聚合，不是模拟器完全无源。修复后必须生成新版本数据，不能回填旧张量并声称原始协议已有该字段。
- `rate_sum`与`served_data`不应互相冒充：前者应在v4中重建为已分配RB上的链路服务能力之和，后者是受剩余数据上限截断后的实际交付量。二者分别对应无线容量与业务服务结果，保留两者有明确非冗余语义。AirFogSim源码虽然把rate注释为Mbps，但当前直接用`rate*simulation_interval`与未标单位task size比较，尚无可核验bit/byte换算，因此在P2单位校准前只能称`AirFogSim data-unit/s`和`AirFogSim data-unit`，不能写成Mbps与MB。
- 源`source_rb_actions`和transfer event保存具体`rb_indices`，但`airfogsim_tensor_v2.py`当前只把RB动作编码成`task_action`中的动作标志、`rb_count`和归一化count；没有保存RB身份/bitmap。由于同频干扰取决于具体RB重叠，仅有count不足以支持理论中的逐RB干扰/SINR递推。
- 处置边界：RB bitmap/索引属于独立action contract，不应伪装成信息边状态维度；P2新数据协议必须保存它，P4规则层据此更新RB占用和冲突。否则即使补出SINR/干扰标签，动作条件预测仍缺失关键因果输入。
- v4建议采用两种分辨率：链路级核心张量供默认编码器使用；逐RB信道/结果辅助张量与稀疏RB动作独立保存。这样精简默认输入不丢失候选动作RB身份，完整逐RB张量可用于规则层、无线增强编码器和上界诊断。
- 旧R5/R6 artifact能证明运行时间、链路活动/速率、任务时延等指标已有记录入口，但属于旧v3协议，不能用其数值倒推v4字段选择门槛。v4必须事前冻结配对非劣门、效率门和缺失鲁棒门。
- v4自审后的权威动作表示为稀疏`(time,flow,edge,RB)` COO；记录存在即表示二元分配为1，空COO表示全0。恒真`assignment_value`和含混的通用`action_present`已删除，卸载动作另设任务级有效Mask，避免多套存在性语义。
- AirFogSim逐RB结果并不自动意味着未激活RB也有可训练物理语义；未激活RB上的零信号、钳位SINR或零rate必须先经计算路径和样例证明，否则标为无效而不是合法零值。
- v4 registry必须逐字段声明适用边类型：无线衰减/SINR/干扰/outage在wired边属于`not_applicable`；理论上适用但当前源未采集属于`source_absent/not_collected`。wired容量需要自己的直接事件或配置语义校准，不能套用无线RB公式。

## 2026-08-12 P0 新增机器核验

- `trajectory_index.csv` 可直接证实 train 36 / validation 12 / calibration 6 / locked_test 6，共 60 条轨迹；本次只读了索引元数据，没有读取 locked test 轨迹内容。
- 对完整索引聚合后证实：`deadline_aware`、`equal_share`、`feasible_exploration` 三种 CPU 策略各 20 条；六种负载×密度场景各 10 条；60 条轨迹均为 300 个 observed steps。PPT 第194页的这部分数量主张成立。
- R5/R6 正式产物实际位于 `代码/artifacts/formal_training/` 和 `代码/artifacts/analysis/`，不是先前尝试的 `artifacts/experiments/`。两处目标路径不存在是路径假设错误，不是证据缺失；已转向真实目录。
- R5 `candidate_freeze.json` 明确限定 `freeze_scope=R6_working_candidate_set_only`、`final_method_frozen=false`、`locked_test_accessed=false`；因此 R5.1 的“冻结”仅是进入 R6 的工作候选选择，不是最终模型/方法冻结。
- R6 `matrix_summary.json` 证实 18/18 个 10k run 完成、0 失败、未访问 locked test，但同时明确 `target_environment_steps=10000`、`formal_budget_complete=false`；PPT 第202页可表述为“10k 阶段矩阵完成”，不可表述为 100k/正式预算或最终策略完成。
- R5 的真实组合构造在 `r5_world_model.py` 中是可组合的：先用 `_ExplicitDAGBackend` 或 `_SoftPredictedPresenceBackend` 作为 base，再包 `_GraphRSSMBackend`，最后可包 `_HeteroscedasticBackend`；C/D/E正式checkpoint的参数键也分别包含RSSM与异方差/DAG/soft-presence相关参数。因此组合命名在R5路径上有代码与checkpoint依据，不能仅按R4工厂的互斥`elif`推断为错误。
- `normalization_stats.json` 对 18 维信息边状态给出非零 count 的索引为 0、1、8、11、12，共 5 维；索引 2–7、9–10、13–17 的 count 为 0、scale 被置为 1。与此同时，图构造器把 `interface_available` 固定为 1，并把缺失字段填 0 且 mask 为 False；因此当前事实是“18 槽位契约 + 5 维观测有效”，不是“18 维真实信息已补齐”。
- 信息边的 5 个有效量中，`pre.interface_available`、`pre.csi_mean`、`action.allocated_rb_count`、`outcome.active_task_count`、`outcome.rate_sum` 有数据；`csi_mean`来自 AirFogSim channel manager 的 CSI 快照，`rate_sum`为信道管理器返回的各信道速率求和。`channel_gain/path_loss/noise/interference/SINR/outage/throughput/served_data` 当前没有直接有效源；它们被明确 mask，不可改名冒充已测量。
- 代码和测试证实 Graph-RSSM 候选确实有上下文 prior/posterior、KL 辅助项、部署 rollout prior-only 和动作敏感性（`test_r4_rssm.py` 4 项均通过）；但尚无 latent 不塌缩、概率校准或多候选规划证据，所以只能称 RSSM 工作候选，不能称完整概率规划方法。
- 更严格的条件语义核查显示，`rssm_context_prior`与`rssm_context_posterior`都读取同一个当前`base_belief.joint`；现有实现没有证明讲稿所述“prior只看过去、posterior再看当前观测”的分工。因此RSSM证据状态是partial，不是完整实现。
- R6 在线状态路径调用 `model.infer_belief`、显式状态编码和候选描述符，`CandidateMaskedActorCritic.forward` 对候选直接算 logits；未调用 `model.rollout`。它是 belief-conditioned direct candidate policy + 真实执行反馈闭环，不是世界模型候选-rollout planner。10k 产物还显示 normalized selection entropy 约 0.999、非默认选择率约 0.82–0.83，策略尚未形成低熵稳定偏好。
- 针对 R4/R5/R6 核心接口的 32 个单元测试通过；这是契约与前向/反向证据，不等于理论闭环或最终性能已通过。
- 对 54 条非锁定轨迹的 `information_edge_feature_mask` 逐文件复算：18 维中仍仅 0/1/8/11/12 非零，累计有效计数均为 5,854,752；这排除了“只是 normalization 汇总遗漏”的解释，确认缺失发生在实际张量层。
- R2 的 43 项是 registry/协议定义，不是 43 项均已有模型结果。当前 bundle 的机器事实为：43 项注册；22 项事实映射（16 direct、5 alias、1 semantic alias）；zero/last baseline 记录中 1,675 computed、485 not_computable；统一 evaluation rows 中 2,647 computed、647 not_computable、54 not_applicable。旧文档“43项评价协议完成”可保留为协议冻结，但不得缩写为“43项结果全部输出”。
- R1 顶层 manifest 声明 278 个文件；本轮逐文件重算 SHA-256 为 0 不匹配。这个结论只证明现有 v3 bundle 自洽，没有证明字段物理语义正确或 13 个缺失槽位已补齐。
- 双图结构层已有可验证实现：以 seed 0 为例，graph validation 明确通过 CIP 唯一、CEP 端点一致、CFL 端点一致、物理边仅空间关系和 DAG 无环；计数为 22 个物理/信息节点、462 个物理/信息边、173 个流、76 个 DAG 边。其余 seed 的顶层 manifest/hash 自洽，但 P0 尚未逐 seed 重算全部结构语义；因此可称“结构验证产物存在且 bundle 自洽”，不把 seed 0 例子泛化成未检查的逐 seed 证明。
- 主文档把核心动作固定为卸载+RB，CPU为确定规则/可选扩展；R6 `JointActionCandidate` 却同时绑定 offload、RB、CPU 并实际改变 CPU。这是明确的动作空间冲突。R1/R3 的 `task_action` 为 8 维并包含 CPU 标记/分配/比例，说明旧世界模型训练数据也已含 CPU 动作条件；因此不能简单把 R6 CPU 说成无关附加量。P0 处置应先冻结二选一：要么扩展理论核心动作并重新定义可行域/实验，要么从正式规划候选移除可控 CPU、按固定规则执行。未决前阻止 P4/P6 扩训。
- 主文档要求 `U^det` 在每步 rollout 严格更新卸载写入、RB计数、阶段合法转换、路径与守恒；当前 R3/R4 world-model rollout 仅对 latent 做 GRU、图更新/耦合和 learned heads，源码中没有该确定性规则层。当前“规则约束递推”判为 not-implemented；现有 R3-R5 数值仍可作为纯学习 rollout 历史结果，但不能支持该理论贡献。

## 2026-08-12 P0一致性审计

- P0不继承旧计划中的“complete/ready”作为事实；它们只作为待核claim。
- 已由机器可读v3契约与归一化统计复算：信息边字段宽度18，只有索引0、1、8、11、12的有效计数非零，共5维；其余13维不能描述为真实数据已完成。
- 已由R6核心代码确认：在线状态路径调用世界模型`infer_belief`，策略器以当前显式/隐式状态和候选描述符直接计算logit；核心R6文件没有逐候选`model.rollout`调用。因此现状是belief-conditioned direct candidate policy加真实执行反馈闭环，不是世界模型候选Rollout规划器。
- 本轮尚未确认PI-JWM 8.11原始PPT源文件是否存在；在完成文件清单前不对“PPT与实现一致”作判断。
- 材料清单已纠正：已找到`文档/组会/PI-JWM_组会汇报.pptx`（18,971,202 bytes）、`2026-08-11-PI-JWM技术细节与答辩问答.md`、`2026-08-11-PI-JWM组会汇报讲稿.md`和版式标准，因此PPT可直接核验，不需要由讲稿反推。
- 知识库`PIJWM主文档.md`和`8.12之后推进.md`在2026-08-11 23:47/23:52仍有磁盘改动；P0必须重读当前文件，不复用先前内容摘要。
- PPT模板检查辅助脚本因Windows缺少兼容`unzip -Z1`而不能完成媒体包扫描；artifact-tool workspace已成功初始化。P0将直接用artifact-tool导入、结构化inspect和逐页渲染，足以核验页面主张，但不会声称完成了模板媒体/字体审计。
- artifact-tool已成功完整导入`PI-JWM_组会汇报.pptx`并输出203页结构化inspect、逐页PNG/layout和montage。该文件显然包含历史汇报合集，不能默认203页均为8.11当前口径；需结合页标题和8.11逐页讲稿界定有效区段。
- 渲染日志有4次`autoRouteConnectorPx`找不到连接对象并使用fallback path；文本与页面输出完成。该问题只作为预览连线外观限制记录，不用于否定或证明方法主张。
- PPT目录表明第190页是`2026-08-11`分隔页，第191—203页恰好对应讲稿所称13个正文页面；第1—189页为历史汇报合集。P0把191—203作为8.11当前PPT口径，历史页只用于追踪旧表述。
- 8.11技术问答明确设有“当前真实实现到底有几维信息边”“当前实现与完整目标的区别”“在线闭环正确顺序”“Graph-RSSM只是工作候选”“10k正确解读”等边界章节；这些文字仍需与代码/产物逐项核验，不能因文档自我限定就自动判真。
- 自写的页级inspect文件未被artifact-tool按`slide.id`收窄而重复包含全deck并发生截断，不能作为页级证据；完整deck NDJSON及按slide对象导出的203张PNG有效，后续只使用这两类来源。

### 8.11 PPT有效区段的直接主张

- 第191页声称：54条非锁定轨迹、15,660窗口、43项指标，三类模块实验完成，已跑通新双图训练与真实在线闭环；100k、最终定型、baseline未完成。
- 第193页把信息边列为18维完整协议，但页面没有同时显示其中只有5维真实有效。这会让“接口宽度”和“有效数据”混淆，必须在P0判为表述缺失。
- 第194页称60条轨迹、每条300时隙、36/12/6/6划分、3类CPU策略每类20条；还称“显式生成代理附着、信息边承载和信息流映射三类跨图关系”。这些均待与manifest/张量数组核对。
- 第195页展示43项四层指标，但页面措辞“完整评价指标体系”不等于43项均已有可计算结果；需核对R2协议、当前评价产物的available/not_computable状态。
- 第196页明确画出目标闭环：生成合法动作→世界模型并行预测各候选→按KPI/约束/不确定性选择→执行首动作→更新历史重规划，并把Graph-RSSM称为“当前主候选”。
- 第201页实际策略定义是Masked Actor–Critic/PPO直接在候选集合上输出动作概率，输入显式/隐式状态；没有候选世界模型rollout步骤。第196与201页因此存在目标机制和当前实验机制未分层的冲突。
- 第202页三种子10k表原始值为：Actor-Critic显式`0.199±0.006/0.492`、隐式`0.217±0.020/0.487`、联合`0.224±0.031/0.474`；PPO显式`0.198±0.004/0.492`、隐式`0.210±0.010/0.485`、联合`0.206±0.001/0.503`；六组均写100%按时完成和0硬约束。必须验证这些表值的源artifact、分母和指标语义。
- 第198页四组合表和第200页五世界模型表已通过原生table API精确读取，未用截图猜数；其数值将与R5正式分析artifact交叉核对。
- 第203页原计划先补信息边再100k；P0继续维持100k阻塞，但“补全”必须改为逐字段判定direct/derived/fixed_config/unavailable，而非追求18维全开。

### P0 Evidence Policy

- 证据等级固定为：verified、partial、interface-only、contradicted、not-implemented、not-verifiable。
- 每个verified结论至少要有可定位代码/数据/测试/artifact中的适用证据；理论条件命题不能替代工程证据。
- 任何关键contradicted或not-verifiable项都会阻断旧协议扩训和最终方法冻结。

## 2026-08-08 R6.1设计冻结补充

- 正式策略动作采用“完整可行联合方案候选集”：每个候选同时绑定卸载、RB和CPU，候选0固定为AirFogSim默认合法调度。策略只在统一候选集和mask上选编号，避免逐任务三头动作冲突；候选集是受控子集，不宣称穷举全局动作空间。
- 显式状态与Graph-RSSM隐式belief必须并存，并冻结`explicit-only`、`latent-only`、`explicit+latent`三种状态消融；未来target、跨split窗口和locked-test均不得进入策略输入。
- 正式即时reward只使用可按transition归因的按期完成、新增失败、完成时延增量、交付量增量和能量增量；P95/P99、Jain公平性、资源利用率、action regret和不确定性保留为episode/系统评价指标。
- reward尺度只由36条train系统目标轨迹估计，validation/calibration/locked-test不参与；硬约束违反使transition无效，不能用负reward抵消。
- 首轮GPU矩阵冻结为Masked Actor-Critic/PPO × 三种状态 × 三个seed，共18个正式运行；本阶段只完成协议、CPU实现和真实预检，不启动GPU或宣称策略性能。
- 本地Zotero检索未发现PPO和GAE原始条目；并行3查询曾因外层24秒上限超时，改为单查询60秒后得到明确“未找到”结果。后续如写入文献库，必须按PIJWM根/子集合与标签协议去重导入并回读验证。
- 正式系统目标字段`delivered_data_total`经seed 0原始数组核对不是累计曲线，而是每时隙交付量：数组会从正值回到0，300步求和为47.137182880968254 MB并与轨迹报告一致。R6 reward直接使用该逐步量，不能再次差分；首次累计解释已被正式尺度冻结的非单调检查拦住并修正。
- 正式reward尺度已从36条train轨迹冻结：完成时延P95 `1.699999988079071 s`、每步交付量P95 `1.3240530347144954 MB`、能耗增量P95 `17.637960052490236`；validation/calibration/locked-test均未参与尺度估计。
- 正式AirFogSim闭环必须同时复用`EvidenceLoggingAlgorithm`的DAG-ready调度语义、Observed通信路径和非变异传输任务查询。只复用普通`BaseAlgorithm`或标准`AirFogSimEnv`会分别触发卸载断言或造成冻结/实时任务数偏离，不能作为同轨迹策略训练环境。
- validation seed 507时隙80—83已产生4条连续真实transition；独立非空扫描实际改变卸载1次、RB 2次和CPU 1次。动作台账保存具体任务—目标、RB编号和CPU数量，因此`real_nonnoop_action_evidence`不是候选结构或no-op证据。
- Actor–Critic/PPO各一次CPU更新有限，冻结候选B世界模型参数哈希前后均为`4f0f05216917bcf4a34a0c00beddb3aac6ee6b1629ca1f65357a5cdf4ddbd6a0`。这只验证训练入口，不支持性能、收敛或最终方法结论。
- 首轮GPU矩阵已冻结为2方法×3状态模式×3 seed共18个正式run；MPC不进入首轮，失败seed保留且不得替换。九项CPU前置门通过，当前可置`r6_gpu_strategy_training_ready=true`但仍为`final_method_frozen=false`。

## 2026-08-08 R6.1 GPU训练前协议初始审计

- 已完成的R6学习策略CPU门只在真实冻结状态上训练连续CPU动作，卸载和RB为安全no-op；不能直接把该smoke扩展成GPU联合策略训练。
- `formal_airfogsim_runtime_v1.py`目前公开的注入点是`allocator_factory`，只替换计算调度CPU回调。正式联合动作需要先确认已有卸载/RB调度辅助和执行时序，再设计薄适配层，不能在runtime中凭空创建第三方接口。
- 现有`airfogsim_diagnostics.reward_components()`只覆盖完成/失败/吞吐量的历史诊断重构，未覆盖R2冻结的尾时延、能耗、公平性和硬约束三态语义；R6.1需要独立、版本化、机器可读reward协议。
- R6.1应复用R1/R2/R5.1/R6资产，不重新生成数据或重训世界模型；阶段终点是GPU入口门和CPU证据，不是策略收益。
- `run_formal_airfogsim_trajectory()`通过临时替换`install_capacity_safe_cpu_callback`注入CPU分配，运行后恢复原函数；它没有卸载/RB策略注入参数，因此联合动作不能只扩展`allocator_factory`的返回值。
- 已有卸载/RB/CPU统一候选执行入口位于`代码/scripts/run_pi_jwm_energy_reward_diagnostic.py::prepare_candidate_step`，相关测试已覆盖默认调度先执行、延迟卸载、RB裁剪和CPU容量裁剪。R6.1应提取/复用其合法机制，而非复制一份不同语义实现。
- 规则配对CPU投影已经按节点容量工作，但只接收任务权重映射；学习策略的真实闭环适配需要把`ExecutableAction`确定性映射回AirFogSim任务ID、目标节点和RB列表，并保留投影台账。
- 已验证的候选执行顺序是：先用`algorithm.scheduleStep(env)`生成默认合法调度，再在同一时隙覆盖卸载/RB/CPU动作；这避免学习策略绕过AirFogSim原生阶段与依赖过滤。
- 卸载合法候选来自`taskScheduler.getAllToOffloadTaskInfos()`和`entityScheduler.getNeighborNodeInfosById()`；RB总容量来自`commScheduler.getNumberOfRB(env)`，且需要为未选择任务保留默认RB后再从剩余RB中投影；CPU只作用于已经分配并位于执行节点的计算任务。
- 当前causal helper以rank/coverage/scale规则描述动作，不足以表达学习策略逐任务目标节点、RB集合和CPU连续量；R6.1需要一个显式`JointActionBinding`把策略槽位映射为任务ID/节点ID/RB编号，随后复用相同执行顺序和公共调度API。
- 现有`apply_offload_overrides()`只排除已经计算/正在计算的任务，然后调用`task.changeOffloadTo()`；合法目标仍必须在动作绑定构造阶段由当前邻居候选生成，不能信任任意节点ID。`apply_cpu_overrides()`只覆盖当前已分配且位于执行节点的任务，符合阶段语义。
- R2公平实验协议已经冻结训练seed `20260803/4/5`、失败seed保留、同窗口/同优化步公平、validation选模、calibration只定阈值和R9前locked-test封存。R6.1应继承这些原则，但RL预算必须另以环境步、rollout长度、更新次数表达，不能把世界模型`100 epochs`机械当成策略预算。
- R2 canonical系统指标包括完成率、mean/P95/P99时延、priority-weighted completion、应用吞吐量、RB/CPU利用率、UAV总能耗/单位完成能耗、Jain公平性、硬约束和action regret；正式reward只能选其中可逐transition因果归属的分量，其他指标保留为评价而不是强塞进即时reward。
- R2现有归一化文件只包含双图/任务状态特征尺度，没有系统reward尺度；R6.1不能误用状态标准差归一化吞吐量或能耗，必须从train split真实执行台账另算reward尺度并绑定hash。
- AirFogSim公开接口足以构造逐步事实反馈：`getLastStepSuccTaskInfos()`、`getLastStepFailTaskInfos()`、`getLastStepDoneTaskDelay()`、`getAllTasks()`、当前等待/卸载/计算任务集合、全局channel data和能量快照。无需修改第三方核心。
- `TaskScheduler.getAllToOffloadTaskInfos(env, check_dependency=True)`支持原生DAG依赖过滤；R6.1动作绑定必须显式使用`check_dependency=True`，避免把未释放子任务暴露给策略。
- 现有诊断环境固定`RewardScheduler`为`1/task_delay`，但R6.1不应依赖该内部reward作为唯一训练信号；应由PI-JWM在每个transition上根据冻结协议重建分解reward，并保留AirFogSim原始反馈作审计字段。
- `airfogsim_teacher_aligned_v3`只保存图映射、张量和验证报告，不包含逐步原始能量/吞吐量台账；reward尺度不能从该目录直接恢复。
- 可复用的真实系统结果位于`airfogsim_formal_system_targets_v1`及R6配对闭环`metric_rows.csv`。正式reward尺度应优先由train split事实系统目标或重新执行的train CPU下界轨迹计算，并在协议中记录来源hash；validation/calibration不得参与尺度估计。
- `airfogsim_formal_system_targets_v1`覆盖54个非锁定seed（36 train、12 validation、6 calibration），含8468个完成事件、4983.79 MB交付数据、174818.37能量与32400条能量记录，适合作为reward尺度的事实来源；locked-test未访问。
- R6配对闭环已经在每个策略/轨迹上输出22类系统指标，但ID使用短名（如`task_completion_rate`、`information_throughput`），R6.1协议需提供到R2 canonical ID的显式映射，不能靠字符串猜测。

## 2026-08-04 R4 GPU服务器初始审计

- 新服务器为RTX 4090，显存24564 MiB，连接时空闲显存24081 MiB、GPU利用率0%；`/root/autodl-tmp`可用约46 GB。
- 默认shell没有`python`命令，因此不能直接启动R4；需要先定位既有Conda/venv入口并核对PyTorch CUDA版本。
- `/root/autodl-tmp`存在多个历史PI-JWM目录。这些目录可能复用环境或数据，但语义版本未知；在核对R1 manifest/hash前不得直接作为R4正式输入，也不得覆盖。
- Conda base位于`/root/miniconda3`，Python 3.12中的PyTorch为`2.8.0+cu128`，CUDA可用且能识别RTX 4090；无需新建环境或重新安装PyTorch。
- R2冻结的module-screening预算是max epochs 30、patience 5、training seed `20260803`、batch size 32、相同训练窗口和优化步数。checkpoint不能只按普通validation loss选，正式入口必须计算四项等权`validation_protocol_score`；缺任一公共项时该候选不合格。
- 既有`run_formal_dual_graph_gpu_train_v1.py`提供CUDA搬运、峰值显存、训练曲线和checkpoint模式，但使用旧模型/旧指标接口，只能复用工程模式，不能作为R4训练入口。
- R1正式目录除54个非锁定轨迹张量外，还包含`window_index.csv`（约1.1 MB），因此GPU runner应以冻结窗口索引构建训练/验证/校准样本，而不是用CPU预检的`select_r3_windows()`每轨迹只抽一个窗口。
- v3协议再次确认无线信道及CSI、增益、干扰、SINR、RB、速率和吞吐量均属于信息图语义；物理边只保留同一时隙空间关系。GPU实现不得复用旧formal-v1把无线量放在物理边的模型。

## 2026-08-04 R4 CPU实现与真实协议发现

- 冻结R1张量中的普通物理/信息边端点采用成对索引，未占用槽允许`(-1,-1)`；图候选只能对当前活动边强制合法端点，不能把合法padding误判为坏数据。R-GCN和edge-conditioned候选现已统一执行该规则。
- 正式DAG端点不是时间序列，而是静态`(B,2,E)`数组：第0行是父任务，第1行是子任务；`dag_edge_present`才是`(B,H,E)`动态存在掩码。R4内部转为`(B,E,2)`后传播，仍只用历史presence，不读取未来DAG target。
- `hurdle_active_rate_v1`按R2口径把链路活动分类与活动条件正速率回归分开；若窗口没有活动且可观测的正速率样本，辅助NLL明确记为`not_computable`，不伪造0，也不让整个候选失败。
- Graph-RSSM部署滚动只走动作条件prior；异方差与hurdle概率参数只追加到公共输出，不删除冻结的显式状态、离散logit和隐式belief接口。
- CPU矩阵证明12个受控候选在真实协议上可训练、可滚动、可封存和可复现，但小窗口单步更新不能比较模型好坏。下一步必须按冻结的同窗口、同预算、同seed规则进行GPU短预算筛选。

## 2026-08-04 R4候选方法文献门——初始边界

- R4不再从“先实现一个看起来合理的网络”开始，而是先建立六类方法的证据矩阵：字段编码、关系图编码、双图耦合、动作条件潜在动力学、稀疏/不确定输出、DAG与动态拓扑。
- 已冻结的是PI-JWM科学对象和接口：严格物理图/信息图、`CIP/CEP/CFL`显式关系、显式状态与隐式belief并存、动作条件开放环rollout、R1数据与R2指标；具体编码器、耦合器、动力学和输出头仍是待实验变量。
- 每个候选必须对应一个单独可证伪问题；不做全排列，也不把论文中面向视觉、控制或一般图数据的方法不加审计地直接移植到PI-JWM。
- R3保留为公共参考点。DAG边消息传递和预测presence参与后续拓扑递推是R4的两个明确缺口，不得在调研完成前默认采用某种实现。

### 第一批原始来源核验：世界模型动力学

- PlaNet（ICML 2019）是RSSM和latent overshooting的直接原始来源：确定性记忆与随机潜变量共同描述隐状态，动作条件转移用于多步开放环预测。它支持“随机RSSM是R4动力学候选”，但不规定PI-JWM的图编码和双图耦合。
- DreamerV3的正式版本为Nature 2025《Mastering diverse control tasks through world models》（DOI `10.1038/s41586-025-08744-2`）。其世界模型、actor和critic并行训练属于完整控制框架；R4当前只可借鉴其稳健的表示/损失和RSSM，不应把actor-critic混入结构筛选。
- TransDreamer的OpenReview记录为ICLR 2022撤回稿，证据等级明显低于PlaNet/DreamerV3。它说明Transformer状态空间模型可提供长程记忆，但不能作为“必须采用Transformer”的正式高等级依据；在PI-JWM数据规模和序列长度证据不足时只能作为后置候选。
- TD-MPC2在OpenReview有正式论文记录，需继续核对SimNorm、潜在一致性目标和模型预测控制各自作用；其中MPC属于后续策略/规划阶段，不能作为R4状态模型结构的一部分。

### 第二批原始来源核验：图编码

- R-GCN（ESWC 2018，DOI `10.1007/978-3-319-93417-4_38`）为多关系图定义关系类型专属变换，并用basis/block decomposition控制参数量。它适合作为`CIP/CEP/CFL`及分支内离散关系类型的清晰基线，但原论文面向知识图谱，不能直接说明如何处理PI-JWM的连续信道、队列和空间边属性。
- ECC（CVPR 2017）由边标签动态生成卷积滤波器，直接支持连续边属性。它对信息边上的信道特征和物理边上的空间特征更贴合，但原始任务不是时序无线系统，必须保留relation type并验证参数稳定性。
- GATv2（ICLR 2022）修复标准GAT的静态注意力排序限制，在相同参数量级提供query-conditioned动态注意力。它支持“邻居重要性应随接收节点状态改变”的候选假设，但原论文不证明注意力优于显式关系/边条件消息传递，不能默认替代`CIP/CEP/CFL`语义。
- GNS（ICML 2020）提供encode-process-decode图动力学先例，并指出消息传递步数和训练时噪声对长rollout重要。它最适合作为PI-JWM关系消息传递的通用机制依据，但粒子图不是物理—信息双图。
- Shen等（IEEE JSAC 2021，DOI `10.1109/JSAC.2020.3036965`）直接支持无线系统的节点/有向链路图和置换等变GNN。其证据用于确认信息图通信边与可扩展图编码，不用于证明某个R4网络结构必胜。
- 因此图编码候选应保持可解释层级：**R3关系MLP/GNS式消息传递（参考）→ R-GCN（离散关系基线）→ edge-conditioned relation MPNN（连续边属性候选）→ GATv2式动态权重（注意力候选）**；每次只替换分支内图算子，不同时改变耦合和动力学。

### 第三批原始来源核验：双图耦合与图世界模型

- Bou Chaaya等（IEEE TWC 2026，DOI `10.1109/TWC.2025.3644600`）确实采用两个耦合JEPA：控制JEPA的潜在表示通过cross-modal conditioning指导无线JEPA预测，并用ensemble估计不确定性、MPC做调度。它是PI-JWM“分支独立编码＋潜在跨模态条件”的强直接先例。
- 该TWC方案的耦合是**有方向且场景特定**的：控制状态指导CSI动力学；它没有物理/信息实体一一附着、`CIP/CEP/CFL`三种显式关系，也没有任务DAG、队列、卸载和双向耦合。因此它支持R4的“方向性JEPA耦合”候选，不能直接固定为PI-JWM最终耦合器。
- Graph World Model（ICML 2025）展示两条统一路线：把多模态转为统一token，或使用模态专属编码器进入统一embedding空间，再通过消息传递聚合结构信息。PI-JWM更符合后者，但该论文把任务表示成action nodes，与PI-JWM已冻结的“任务/DAG不是信息图节点”语义冲突，不能直接复制其图对象。
- 2026年G-RSSM预印本为每节点保留GRU与随机潜变量，并用跨节点多头注意力建模无线网络中的移动、能量和拓扑变化；它直接提示“每实体latent而非整图压成单向量”。但其证据等级是最新预印本，图中也没有严格物理—信息双层语义，只能作为R4候选而非顶刊支撑。
- R4耦合应至少比较：无耦合、结构约束门控残差（R3参考）、只沿真实`CIP/CEP/CFL`的cross-attention、方向性JEPA条件。随机打乱跨图对应关系是因果/结构对照，不是新方法。

### 第四批原始来源核验：不确定性、动态拓扑与DAG

- Deep Ensembles（NeurIPS 2017）支持独立初始化的概率网络集合用于回归/分类预测不确定性，并强调NLL/Brier等proper scoring rule和分布移位评估。它适合作为R5正式不确定性层；若在R4使用，只能在基础结构筛选完成后检验稳定性，不能让多倍训练预算干扰结构公平性。
- Kendall与Gal（NeurIPS 2017）区分输入相关aleatoric不确定性和模型epistemic不确定性，支持连续输出同时预测均值/方差的异方差头。对PI-JWM而言，单模型异方差头可进入R4输出头筛选；ensemble用于后续模型不确定性。
- VGRNN（NeurIPS 2019）联合建模动态图拓扑和节点属性变化，证明“拓扑本身可作为随机预测对象”有成熟先例。但它面向动态链路预测，不满足PI-JWM的信息边物理端点、接口兼容和协议约束；因此presence预测必须先经规则合法性壳，不能自由生成任意边。
- 多DAG卸载文献（IEEE IoT-J 2021，DOI `10.1109/JIOT.2020.3030926`）把子任务依赖和网络流调度联合建模；IEEE TC 2022（DOI `10.1109/TC.2021.3131040`）用DAG表达依赖任务；IEEE JSAC 2023（DOI `10.1109/JSAC.2022.3233532`）围绕DAG前驱、在线到达和deadline violation建模。这些文献直接支持DAG是任务辅助结构、DAG消息应影响ready/进度/时延预测，而不支持把DAG边当成信息通信边。
- R4的输出/结构递推候选应分开：连续状态的确定性头 vs 异方差头；零膨胀链路量采用activity分类＋active-only正值回归的hurdle分解；拓扑采用固定真值掩码/预测soft presence/阈值hard presence三种递推方式；DAG采用三维摘要/显式DAG消息传递。不同问题不能混成一个“大改模型”。

### 第五批原始来源核验：字段表示与稳健潜变量

- DreamerV3正式版本明确：向量输入用symlog变换抑制大数值和大重建梯度；随机latent采用多组categorical分布；dynamics/representation KL采用stop-gradient分工、free bits和不对称权重以避免无信息塌缩。它们是不同技术，不应被笼统称为“归一化”。
- TD-MPC2（ICLR 2024，arXiv `2310.16828`）采用decoder-free隐式世界模型、latent一致性、SimNorm、离散回归和Q ensemble，并把MPPI规划用于连续动作。对PI-JWM可独立借鉴的是**SimNorm/latent一致性**；Q函数、actor和MPPI属于策略/规划，不进入R4状态模型筛选。
- SimNorm把latent按组投影到概率单纯形，目标是限制表示尺度并改善可扩展训练；DreamerV3的symlog作用在数值观测/目标尺度，两者作用位置不同，可以形成“train-only标准化参考、+symlog、+SimNorm”三档字段/latent候选，不能同时加入后把收益归因给某一个。
- PI-JWM已有按字段train-only归一化和mask，因此R4字段实验不应重写数据协议。候选只允许在编码器内部对已归一化有效值加入symlog或分组SimNorm，并保持缺失掩码、单位和显式状态原值不变。

### TWC耦合JEPA的可复现机制与边界

- 控制JEPA实际是RSSM：`h_t=f(h_{t-1},a_{t-1},z_{t-1})`，posterior读取当前编码观测，prior只读历史hidden；训练损失由KL、reward NLL和termination NLL构成，并用两路stop-gradient KL balance防塌缩。它不是单纯的“两个encoder做相似度”。
- 无线JEPA将CSI编码为`c_t`，预测器以`c_{t-1}`、无线隐状态以及控制latent `(h_t,z_t)`为条件预测未来`c_t`，用多步latent L2损失；target encoder由EMA更新，训练无线JEPA时控制JEPA冻结。这是一个明确的**先控制、后无线**分阶段方向性耦合方案。
- 论文假设每个设备运动只显著影响自身CSI、设备任务相互不相关，并为每个agent单独训练网络。PI-JWM包含干扰、共享RB、跨代理任务流与DAG，不能接受这些独立性假设；若做JEPA候选，必须改为共享参数的图级分支，并沿真实跨图关系进行条件传递。
- 论文ensemble为5个不同初始化/随机批次的latent transition MLP，以预测方差衡量分歧；它适合后续不确定性对照，但其累计不确定性和MPC调度属于策略层，不属于R4表示结构。

### R4文献门第一版结论

- 首轮候选已压缩为：字段`Masked MLP/symlog/SimNorm`；图编码`Relation-MPNN/R-GCN/edge-conditioned relation MPNN`；耦合`无耦合/门控/cross-attention`；动力学`Graph-GRU/Graph-RSSM`；输出`typed deterministic/heteroscedastic/hurdle`；结构`DAG summary/DAG message passing`与`fixed/soft presence`。GATv2、方向性JEPA和hard presence为第二层候选。
- Transformer、diffusion dynamics、完整Dreamer actor-critic、TD-MPC2 MPPI/Q、Deep Ensemble、foundation pretraining和跨场景迁移均不进入R4首轮；分别推迟到出现明确长依赖瓶颈、R5不确定性复验、R6/R7策略与论文baseline、或R9之后。
- 每个候选必须只改变一个模块并回答一个可证伪问题；单项最优不自动拼接，R5只复验少量兼容组合。
- 调研正文已固化到`文档/研究进展/2026-08-04-PI-JWM-R4候选方法文献调研与实验门.md`。Zotero新增八条证据已云端回读验证；本地API尚未同步，状态为“云端完成、本地等待同步”，不得重复导入。

## 2026-08-01 正式双图模型本地阶段初始发现

- 现有`airfogsim_smoke_model_v2.py`已经提供action-conditioned多步状态预测、存在性头和mask-safe损失，可作为接口参考，但此前只在3条开发轨迹上验证。
- 现有`airfogsim_sparse_diagnostics_v2.py`与`run_airfogsim_sparse_event_diagnostic_v2.py`已经实现零活动、末值保持、链路F1/AUPRC、active-only速率误差、信息流/任务存在和生命周期指标；这些口径应复用到正式数据。
- 旧开发结果显示学习臂尚未稳定超过末值保持，因此正式阶段必须同时保留规则baseline与学习baseline，不能只报告新模型绝对数值。
- 正式张量已扩展CPU动作和DAG派生状态，但现有最小模型与窗口数据集尚未确认是否消费`task_dag_state`、动态DAG边和CPU动作；这是新双图模型的首要接口缺口。
- 本地阶段应先验证完整数据加载、前向/反向、统一评价和baseline公平性；长epoch、多seed与结构消融留到GPU。

## 2026-08-01 AirFogSim张量与smoke发现

- AirFogSim配置对象中存在cloud不等于cloud属于当前物理图；三seed的cloud均无wired或无线直接边，必须从节点、代理和节点快照中同时排除，实际节点/代理数为18/16/17。
- 信息流创建时间不能只信调度动作时间。真实轨迹存在“传输事件在2.6秒、卸载动作记录在2.7秒”的顺序差；张量化必须取动作和直接事件的最早证据，并把完成事件所在时隙视为流存在。
- 不存在的信息流状态必须严格为0。仅提供`flow_present=false`但保留总量会造成潜在未来泄漏；现已用测试和校验器共同阻断。
- 统一容量可由当前三seed冻结为18节点、306物理边、216信息流、163任务和422 DAG边，但它只是开发容量，不代表未来正式场景上限。
- 一轮极小模型能完成训练和分split评价，但链路活动F1为0，active-only速率MAE很高，说明稀疏活动事件是当前首要建模难点；物理边“存在性F1”不能冒充链路活动F1。
- 当前正确下一步是建立零活动/持久性/简单时序基线、采用类别不平衡损失并增加任务生命周期预测；在这些基线缺失前，不应把耦合JEPA的复杂度当作首要进展。

## 2026-07-31 AirFogSim新双图重构初始发现

- 权威理论定义已经冻结在`PIJWM推进.md`第二部分：物理图为真实设备—直接信道图，信息图为信息代理—实际信息流图，任务/DAG不再充当信息节点/边。
- 当前AirFogSim信道管理代码支持V2V、V2U、V2I、U2V、U2U、U2I、I2V、I2U、I2I九个方向；当前项目导出脚本只导出V2U、V2I、U2I。
- 当前配置中的cloud不参与无线信道，`wired.edges`为空，因此不能作为本批数据中的有效通信节点或卸载目标。
- 当前普通任务结果数据量为0，且车辆/UAV任务的`p_DAG=0`；新契约必须允许`result_return`和`dependency_data`为空，同时支持未来启用DAG或mission结果后的真实生成。
- 工作区存在大量既有修改与未跟踪PI-JWM文件；本轮只增量修改直接相关文件，不回退用户内容。
- `代码/src/pi_jwm/airfogsim_contract_adapter.py`当前只提供CPU容量安全分配、直接传输能量台账和缺失值编码，没有新双图构造接口。
- 小实验03能够记录逐时隙真实无线传输事件并追踪物理承载，但其`information_nodes`仍是任务记录、`information_edges`仍是DAG边，属于已废止的旧信息图语义；可复用的是运行时事件记录、路径合并、守恒和产物写出机制。
- 冻结exp03产物包含19个物理节点、97条物理边、159条真实传输事件、163个旧“信息节点”（实际为任务）和422条旧“信息边”（实际为DAG先后关系）。
- exp03通过`build_preflight_config()`把车辆/UAV的`dag_edge_prob`提升到0.6，并记录任务输入与结果回传，因此可作为启用DAG后的原始样例来源；但DAG边的`data_mb`仍为`null`，AirFogSim只直接提供先后关系。
- 现有`dependency_flows`把父任务结果回传到任务源解释为共享父输出，不能直接当成“父任务执行代理到子任务执行代理”的`dependency_data`信息流；新v2必须保留证据边界或补充明确的PI-JWM扩展规则。
- exp04已有任务通信/计算台账、RB台账、依赖台账和UAV能量台账，可复用为真实指标输入，但其依赖台账同样基于旧共享返回语义，需要在v2中重新标注。
- AirFogSim原生DAG由`TaskManager._generateTaskDAG()`随机生成，只用于判断父任务是否完成/失败以及子任务何时从`to_generate`进入系统；DAG边没有载荷大小，也没有父执行节点到子执行节点的独立传输过程。
- 因此“加入DAG”应分成两层：原生DAG作为可直接观测的任务先后结构；跨代理`dependency_data`必须由PI-JWM显式扩展契约定义载荷和传输，否则只能标为`not_modeled`，不能把返回流改名冒充。
- 现有`main_experiment_contract.py`仍把任务身份/生命周期列为信息节点，把DAG和共享父输出列为信息边，并把CPU分配列为核心动作；它属于旧契约，必须在v2结构产物通过后修订。
- AirFogSim任务对象直接提供到达时间、deadline、优先级、输入量、计算量、结果量、已传输量、已计算量和各阶段最后时间；`TaskManager.getAllTasks()`包含等待、传输、计算、回传、完成和失败任务，因此逐时隙运行器可以补齐真实任务结果字段。
- 新`airfogsim_dual_graph_v2.py`已将信息节点固定为`agent::<physical_id>`，把真实offload/return事件分别映射为`task_input`/`result_return`，把DAG保留在`task_dag_edges`，并生成CIP/CFE关系。
- v2依赖规则已经测试：`data_mb=null`只保留DAG；只有显式正载荷且父子任务执行代理不同才生成等待中的`dependency_data`，等待流允许暂时没有CFE。
- 第一版v2重构产物已能形成19个物理节点、97条物理边、19个信息代理、158条真实信息流、163个任务节点、422条DAG边、19条CIP和158条CFE；其中86条`task_input`、72条`result_return`，没有伪造`dependency_data`。
- 冻结exp03包只保存每个物理节点/边的末次状态：Task_10信息流发生于`t=2.6`，对应物理边末次状态为`t=2.9`，节点末次状态为`t=12.0`，因此它只能支撑轨迹级结构重构，不能作为严格单时隙样例。
- 现有60-seed原始数据含逐时隙`node_states.csv`和`link_states.csv`，但其任务配置明确为`dag_edge_prob=0.0`；该数据不能与启用DAG的exp03样例静默拼接。
- 严格单时隙样例需要在启用DAG的同一次AirFogSim运行中同时记录物理节点、物理边、任务/DAG、动作和信息流；计划通过扩展PI-JWM侧实验记录器实现，不修改AirFogSim内核，也不覆盖旧exp03产物。
- 独立exp06同源运行已完成：19个物理节点、306条九方向物理边、19个信息代理、158条真实信息流、163个任务和422条DAG边；73条结果回传事件全部能在同一时刻找到反向物理承载边。
- Task_10样例已严格时间对齐：事件、UAV_0、RSU_3和`pe::UAV_0::RSU_3`均为`t=2.6`。
- seed 0、12秒窗口中163个任务有127个可评价、65个完成、62个失败或到期、36个右删失；完成率0.511811，成功任务P95/P99时延0.8/0.872秒。
- 第一版资源结果为RB利用率0.567167、CPU利用率0.089521、UAV总能耗1186.867696 AirFogSim能量单位；任务流、CPU容量和UAV能量方程违例率均为0。
- AirFogSim原生422条DAG边仍无显式依赖载荷，故依赖载荷覆盖率为0且依赖数据传输率为`not_applicable`；这表示数据缺口，不表示依赖传输性能为0。
- 不确定性覆盖、action regret和OOD迁移需要世界模型分布输出、同状态反事实动作或跨域split，当前真实轨迹评估器正确标为`not_computable`。

## Frozen Evidence

- Local/remote source commit: `f7cb651c57262a1938e7b44ea43c3f3bbee12a44`.
- Formal run: 12 configurations x 3 training seeds x 20 epochs; 36 checkpoints.
- Ranked default validation active-rate RMSE: 233.7162005.
- Sample oracle validation RMSE: 105.3486359 (`sample_oracle`, headroom only).
- All CandidateSet configurations selected the default for every validation sample; nominal RMSE 233.7162005.
- Best classical deployable comparison was GB pairwise at 233.7057583, an immaterial 0.0104 improvement.
- Stage-only diagnostic policy selected on calibration reached validation RMSE 231.3377871 and improved 6/10 seeds. This is the current strongest interpretable diagnostic, not a frozen selector.
- Aggressive HGB pointwise rules overfit calibration and reached validation RMSE 234.5976.

## Current Question

Does the CandidateSet model fail because its candidate ordering is wrong, or because uncertainty calibration and/or the Pareto/defer rules suppress useful rankings?

The candidate-generation gate itself passed strongly on validation: sample-oracle RMSE 105.3486, nontrivial ratio 0.8369, identity oracle-win ratio 0.2575, and action-applied ratio 1.0. This isolates the immediate failure to candidate selection rather than candidate headroom.

## Required Attribution Policies

1. Formal z=1.64 plus Pareto.
2. z=0 plus Pareto.
3. Ensemble variance only, excluding predicted uncertainty.
4. Rank-only plus Pareto.
5. Rank-only without Pareto.
6. Improvement-only argmax.

Each policy must report global RMSE, execution/defer count, realized positive-benefit precision, negative-selection rate, per-seed RMSE, and improved-seed count.

## Formal Checkpoint Attribution

- The 12-config x 6-policy attribution completed on validation only.
- Rank-only plus Pareto produced RMSE 292.90-301.86; rank-only without Pareto produced RMSE 298.82-312.33. Every rank-only policy worsened all 10 validation seeds.
- Rank-only positive-benefit precision was only about 24%-29%, with roughly 47%-54% negative selections. The learned ordering itself is invalid.
- Removing aleatoric uncertainty or setting z=0 executed only a handful of candidates and improved at most 0.067 or 0.025 RMSE, respectively.
- The best improvement-head diagnostic was RMSE 233.5862, only 0.1300 better than default, with 16 active executions and 31.25% positive precision. A more aggressive improvement-head variant improved 7/10 seeds but only by 0.0939 global RMSE and had a 31.45% negative-selection rate.
- Root-cause decision: do not tune the defer threshold. The next method must replace the candidate ordering representation/objective.

## Schema-v6 Interaction Audit

- The formal schema-v6 cache contains 72 x 25 edge-step interaction tokens and 234 pooled interaction features for every sample-candidate pair.
- The completed CandidateSet ranker ignored both arrays and used only the 75 global candidate features plus context.
- A full train/calibration/validation HGB audit compared selected-edge, pooled-interaction-only, and full schema-v6 feature groups.
- Full schema-v6 learned opportunity detection well: validation opportunity ROC-AUC 0.8752 and PR-AUC 0.9495.
- Candidate ranking remained poor: sign PR-AUC 0.4754, sample rank Spearman 0.0832, top-1 positive ratio 0.2825, and no calibration threshold satisfied the safety gate.
- Root-cause hypothesis is now specific: opportunity detection is identifiable, while candidate benefit requires token-level local interaction encoding rather than global or hand-pooled statistics.

## 2026-07-20 Literature Novelty Audit

Search scope: OpenAlex public works API, exact-title and keyword searches covering `world model`, `digital twin`, `UAV vehicular edge computing`, `model-based rollout`, `candidate action selection`, and `energy-aware UAV offloading`; abstracts were checked for the closest papers. This is a documented first-pass novelty audit, not an exhaustive systematic review of every publisher index.

Closest prior work that overlaps the project:

- FlexEdge, DOI `10.1109/TVT.2023.3262261`, already combines a digital twin, UAV-aided vehicular edge computing, joint trajectory/resource optimization, PPO, and energy minimization.
- RADiT, DOI `10.1109/JSAC.2023.3310048`, uses a digital-twin representation of an IoV network and multi-network DRL for task offloading/resource allocation, with energy, delay, and task-completion objectives.
- Adaptive Digital Twin for UAV-Assisted Integrated Sensing, Communication, and Computation Networks, DOI `10.1109/TGCN.2023.3298039`, uses DT prediction plus MAPPO and explicitly balances sensing/computation energy.
- UAV-Assisted Task Offloading in Vehicular Edge Computing Networks, DOI `10.1109/TMC.2023.3259394`, formulates online UAV-assisted VEC offloading with a long-term UAV energy constraint and solves it with Lyapunov/Markov-approximation methods.
- Policy Rollout Action Selection in Continuous Domains for Sensor Path Planning, DOI `10.1109/TAES.2021.3057649`, establishes policy-rollout-based action selection from continuous candidate sets outside edge computing.
- World Models, arXiv `1803.10122`, and Recurrent World Models Facilitate Policy Evolution, arXiv `1809.01999`, establish learned recurrent world models for policy evaluation/evolution in other environments.
- Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities, DOI `10.1613/jair.1.15320`, covers learning models through downstream constrained decision quality; this makes generic decision-focused/listwise claims non-novel by themselves.

Novelty assessment after the search:

1. Not novel alone: digital twin for UAV/VEC, energy-aware task offloading, PPO/MAPPO/DRL resource allocation, graph encoders, latent rollout, Pareto objectives, uncertainty/defer, or candidate ranking.
2. Plausible modeling contribution: a physical-information joint action-conditioned world model that predicts activity, rate magnitude, node/resource, and task state together for UAV-assisted vehicular fog/edge systems. The closest papers found use DT as a system representation or DRL state source; they do not obviously provide the same learned multi-head, action-conditioned rollout interface.
3. Strongest potential contribution: the unified decision protocol that freezes PI-JWM as a rollout evaluator, generates feasible coupled offload/RB/CPU/return candidates, labels candidates by actual rollout benefit, separates opportunity detection from benefit ranking, and applies a task-energy Pareto/defer fallback. However, this is a combination-level claim only. Google Scholar also surfaced recent, highly relevant preprints that narrow the gap: `Agentic World Modeling for 6G: Near-Real-Time Generative State-Space Reasoning` (arXiv:2511.02748, v2 June 2026) already combines action-conditioned generative dynamics, PRB control inputs, uncertainty, what-if analysis, MPC/CEM planning, and offline policy screening; `Vision-Language-Action Models Meet World Models: Embodied Agentic AI for Low-Altitude Wireless Networks` (arXiv:2606.11618, June 2026) explicitly uses a world model for UAV action--environment coupling, policy verification, and dynamic optimization. Therefore PI-JWM cannot claim generic action-conditioned world-model planning or policy screening as unique.
4. Empirical contribution: the controlled finding that true-future action conditioning gives a much lower world-model error than autonomous policy actions, and that the bottleneck localizes to action value/magnitude (especially `rb_total`). This can be a defensible scientific finding if reproduced across independent seeds and baselines, even if the individual algorithms are not new.

5. Revised likely differentiator: the project should focus on the domain-specific physical-information joint representation and the tightly specified, simulator-aligned candidate protocol for UAV-assisted vehicular fog/edge computing: separate activity/rate/task/resource heads, feasible coupled offload/RB/CPU/return actions, actual AirFogSim task-energy audits, per-step reward/energy decomposition, and a deployable selector that never uses future simulator outcomes online. The empirical localization of the bottleneck to resource-action magnitude (`rb_total`) may also be a contribution if independently reproduced. These are narrower and more defensible than claiming a new generic world-model planner.

Required wording for a paper: use “to the best of our knowledge, existing work has not reported [the exact domain-specific combination]” and enumerate every component. Explicitly distinguish prior O-RAN/low-altitude world-model planning from our physical-information UAV-VEC formulation and AirFogSim-aligned audit protocol. Do not write “the first” or “no one has done this.”

## Observable Episode Phase

- Every seed contains exactly 390 consecutive samples, and sample IDs follow `seed * 390 + local_step` on all unlocked splits.
- Current episode phase is therefore recoverable as `sample_id mod 390` without exposing seed identity or future state.
- Adding four phase terms (linear, squared, sine, cosine) to the full schema-v6 HGB audit changed validation RMSE from 233.7162 to 232.0230.
- The phase-aware audit executed six candidates, all six had positive realized benefit, and benefit Pearson correlation increased to 0.5775. This is evidence that task evolution position is a missing deployable context feature.

## Token Selector Probe

- A full-data CUDA smoke completed in 42 seconds and a 3-seed x 20-epoch probe completed in 205.8 seconds.
- The no-phase token model's loss decreased steadily, but calibration candidate sign probabilities did not separate safe actions: threshold 0.50 executed 41 candidates at 43.9% positive precision; threshold 0.65 executed none.
- The no-phase token probe therefore remains diagnostic-only and is retained as a controlled ablation.

## Phase-conditioned Benefit LCB

- Exact within-episode phase is substantially more stable than learned all-candidate ranking. A train-only phase table estimates each candidate's mean raw-SSE benefit, cross-seed positive direction rate, variance, and support count.
- Calibration alone selected the risk/defer rule. Validation selection never changed the calibrated thresholds.
- Enforcing deployable candidate masks and the observable task-energy Pareto gate produced validation active-rate RMSE 207.5399 versus 233.7162 default, a 26.1763 improvement.
- All 10 validation seeds improved. Of 65 active executions, 93.85% had positive realized benefit and 6.15% were negative; Pareto violations were zero.
- Link RMSE improved by 0.574%, and activity F1 dropped only 0.000086.
- The result is B-grade because it is in [200, 213.160874). It passes the general validation safety gate but does not meet the pre-registered <200 A-grade gate, so external seeds 60-69 remain locked.
- Token ranker, candidate-specific experts, learned residual routing, phase smoothing, estimator ensembles, and phase-restricted kNN all failed to improve over the exact phase table. These failed routes are retained as diagnostic evidence rather than hidden.

## P2-C Advisor-Document Manifest Binding

- The previous P2-C canonical manifest bound six source files but omitted the advisor-facing P2-C research-progress document. Therefore document tampering did not invalidate `--verify-only`.
- A RED test demonstrated both failures directly: the portable document key was absent, and a modified project-local copy still verified successfully.
- Adding only the research-progress document to `CANONICAL_SOURCE_PATHS` closed both failures without changing the verifier, audit statistics, candidate formal-data configuration, or status gates.
- The rebuilt canonical differs semantically only in manifest provenance: it adds the portable document key and updates hashes for the modified runner and test. The audit report and candidate config retain SHA-256 values `9c0104d5d589d0c1248663c7e2bbc8739cf4ca3f9241bf53fd0e5efa802c2477` and `6f9b7b7342fe9f326e4216e8dbd073504c0e83fbdbd8b1b7b1a63802f6bf75e0`.
- This evidence closure does not approve formal data: `formal_data_approved=false`, audit status remains `blocked`, and all four pre-existing blockers remain.

## 2026-07-21 Whole-Project Review

### Initial Context

- The root planning files are an active continuation of PI-JWM research, not disposable templates.
- The existing evidence already distinguishes the main world-model line from v11 candidate-selection diagnostics.
- A prior novelty audit identifies the defensible contribution as a domain-specific physical-information joint, action-conditioned rollout model and simulator-aligned evaluation protocol, rather than generic digital twins, graph models, world-model planning, or candidate ranking.
- README content must be re-read with explicit UTF-8 handling because the first PowerShell rendering was mojibake; no semantic conclusion will rely on that corrupted display.

### README and Current Plan

- PI-JWM is defined as an action-conditioned joint transition model over physical-network and information-network state for connected embodied-agent collaboration; AirFogSim is explicitly only the simulator and data generator.
- The implemented main line is: construct physical/information graphs from node, link, task, resource, and action histories; learn multi-step state transitions; predict node/link/task/resource futures; then evaluate autonomous actions with a frozen world model.
- The strongest causal signal so far is that valid future action conditioning materially reduces state-prediction error. The deployable gap is therefore not simply state dynamics capacity; it includes autonomous action support, RB/CPU magnitude reconstruction, and stability of selecting candidates from actual-rollout labels.
- Link-step activity is extremely sparse (about 0.0516%), making support recovery qualitatively different from ordinary dense regression. Support and magnitude should be evaluated separately.
- The evidence protocol is unusually important: deployable, true-future reference, sample oracle, and test-best diagnostic are different result classes; matched seeds 18-19 are already consumed for refinement, while external seeds 60-69 remain locked.
- The 2026-07-20 phase-conditioned benefit LCB result is strong validation evidence (active-rate RMSE 207.5399, 10/10 validation seeds improved, 93.85% positive execution precision), but remains a B-grade v11 selector candidate because the pre-registered <200 gate was not met and external evaluation was not opened.
- The chronological plan contains superseded intermediate statements alongside later outcomes. Source code, frozen summaries, and latest dated evidence must arbitrate current truth.

### Repository Inventory

- Reusable framework code is compact relative to the experiment surface: the core progression is represented by `v6_dual_graph`, `v7_action_policy`/active-rate specialists, `v8_full_world_model`/training, and a family of v11 candidate-labeling, interaction, physical-benefit, selector, and protocol modules.
- The large script/test surface records many controlled attempts. This is useful negative evidence, but also creates a risk that version numbers and diagnostic branches are mistaken for a coherent current architecture.
- Authoritative conceptual documents appear to be `课题介绍.md`, `老师说明.md`, `研究问题与系统边界.md`, `实验结果口径.md`, the two current LaTeX progress/paper drafts, and the dated v11 design documents.
- The first file inventory did not show a tracked `文档/文献` file set despite README describing one. Local untracked/ignored/archived literature locations must be checked before deciding whether external search is necessary.

### Research Question and Boundary

- The formal model question is counterfactual prediction: given a history and a candidate future action sequence, predict the next few node, link, task, and resource states in an air-ground system with UAVs, vehicles, RSUs, and edge nodes.
- The advisor's motivating control problem is broader: online joint communication-resource, computation-offloading, relay, and trajectory decisions under latency-energy objectives, link fluctuation, stochastic arrivals, and partial observation.
- These are not the same contribution. PI-JWM currently addresses the predictive substrate and an autonomous-action interface; a complete online controller with trajectory planning and system-stability guarantees is not yet implemented.
- The repository explicitly separates prediction-world quality from action-generation quality. This separation is scientifically necessary because true-future actions can validate conditional dynamics without demonstrating deployable control.
- Inputs are heterogeneous and causally entangled: physical geometry/topology, information-network link/task/resource state, and future action sequences. Outputs are multi-head state forecasts with optional uncertainty.

### Original Thesis Framing and Evidence Semantics

- The original thesis framing contains two ambitions: a physical-information joint predictor and multi-scenario foundation pretraining with parameter-efficient transfer. Only the first has substantial implementation evidence; the second remains future work.
- Early claims that a joint world model is entirely new or that world models have not been used for network-state prediction are too broad and conflict with the repository's later novelty audit. They should be treated as historical motivation, not current literature conclusions.
- A defensible novelty claim must specify the exact state/action/output combination and evaluation setting rather than claim novelty for generic ST-GNNs, world models, digital twins, or wireless planning.
- The result taxonomy is a core methodological contribution to research hygiene: `deployable`, `true_future_reference`, `sample_oracle`, and `test_best_diagnostic` answer different scientific questions and cannot be compared as if they were equivalent methods.
- Promotion requires a pre-fixed validation rule, reproducible matched evaluation, and non-collapse of auxiliary metrics; isolated small-slice or oracle gains are explicitly insufficient.

### Version Evolution and Paper Narrative

- v0 established the action-conditioned input/output contract but was not a performance win; v1 exposed loss-scale/training issues; v2 added latent recurrent rollout; v3 showed graph context strongly improves activity detection but not active-rate magnitude.
- v4 showed physical features can help but that naive full fusion is seed/threshold-sensitive. v6 provided the first same-split evidence for the dual-graph main line: dual input improved active-rate and link-rate, while physical-only favored node prediction and information-only favored task prediction.
- v7 demonstrated target-specific fusion tradeoffs: the neural model improved active-rate to 206.641 at a cost to activity/task metrics; a random-forest specialist at 92.862 is diagnostic headroom, not the main model.
- v8 tested recurrent latent, STGCN, MoE, and balanced variants. The evidence argues against stacking every sophisticated module: the recurrent option wins active-rate but degrades other heads, while the balanced configuration is the more stable system-level choice.
- v9's hurdle/event formulation correctly decomposes sparse activity from conditional magnitude, but rate-side and activity-side gains remain split rather than unified.
- v10 fixed the future-action alignment problem using the 60-seed action-aligned data and froze the world model as an evaluator. True-future actions reached about 100.14 active-rate RMSE; autonomous policy actions remained around 217.24, localizing the deployable gap to action generation.
- v11 is an evolving diagnostic/decision interface, not the framework's main method. Counterfactual replacement localized much of the error to `rb_total`; actual-rollout candidates have large oracle headroom, but candidate benefit ranking and support remain unstable.
- The current paper draft's most credible contributions are the joint state representation, action-conditioned latent rollout, sparse activity-versus-magnitude evaluation, and controlled frozen-evaluator diagnosis. Claims about a finished autonomous controller or foundation model would exceed the implementation.
- A conceptual concern already visible in the evidence: calling a true-future-action result a "world-model upper bound" can hide distribution and identifiability issues. It proves conditional prediction under logged actions, not necessarily reliable counterfactual prediction for novel action sequences.

### v6 Data Contract and Dual-Graph Implementation

- Physical edge features are deterministically constructed from node histories as relative xyz displacement, 3D distance, speed difference, source/destination speed, and height difference. Information edges come from logged link features; historical and future edge actions are separate tensors.
- Normalization statistics are fitted on train indices, and the dataset supports full, first-step-only, and no-future-action ablations. Outputs are node state, link activity, link rate (including raw targets), and task state.
- The v6 model separately encodes node, physical-edge, information-edge, action, and task histories. Physical/information/action embeddings are fused per edge by concat, gating, modality self-attention, or a hybrid residual attention path.
- Multi-step rollout uses GRUCells: each edge is advanced with its own future action; nodes are advanced from their previous state plus the global mean edge state; task state is advanced from prior task state plus global edge and node summaries.
- Important terminology boundary: the base implementation does not perform explicit graph message passing along an adjacency/incidence operator. Its "dual graph" is a dual-modality edge representation over shared indexed relations, followed by global pooling and recurrent updates. Describing it as a full topology-propagating dual GNN would overstate the code.
- This architecture explains both strengths and weaknesses: per-edge actions can directly affect per-edge latent state, but long-range/local structural effects are compressed through means, which may dilute sparse modifications and obscure task-specific causal paths.
- The v6 helper retains historical default seeds 0-9 in `split_by_seed`, while current scripts/protocols require explicit 60-seed splits. Any new runner must not silently rely on this helper default.

### v8 Architecture and Training Surface

- v8 materially upgrades the v6 skeleton: it imports explicit `DualGraphMessagePassing`, supports history encoders, message-passing versus recurrent latent transitions, sparse adaptive edge context, activity memory routing, hurdle output modes, active-mass allocation, and optional mixture-of-experts rate heads.
- The rollout still preserves the fundamental causal interface: encode observed history, apply each future action at the corresponding rollout step, update graph latents, then decode node/activity/rate/task outputs.
- Training explicitly separates activity loss from rate loss and offers active-only/mixed weighting, focal/BCE activity objectives, hard-negative controls, positive-rate specialists, tail reweighting, hurdle outputs, and auxiliary active-rate heads.
- The large configuration surface is evidence of serious diagnosis, but it also signals method-selection risk. Many mechanisms are experiment branches rather than a single principled final objective; the paper should identify the frozen subset actually used in the claimed model.
- The implemented loss remains a weighted sum of normalized node/activity/rate/task objectives. This is not automatically aligned with downstream global active-rate RMSE or action-selection benefit, which helps explain why improved component losses or heads do not consistently improve the frozen evaluator/selector outcome.
- v8 resolves the earlier terminology issue at the code level by adding explicit graph message passing. Claims about topology-aware dual-graph propagation should be tied to v8, not retroactively to the minimal v6 skeleton.

### v8 Message-Passing Details and Structural Limits

- Each edge update fuses physical, information, and action tokens, then conditions on its source and destination node states. Updated edge messages are aggregated by destination node and degree-normalized before a residual node update.
- The graph topology is a fixed indexed edge set stored in the model config. The implementation validates indices and uses directed destination aggregation; it does not dynamically add/remove topology during rollout, so changing coverage/activity is represented in features/heads rather than changing the graph structure itself.
- Dual-modality separation is strongest at history encoding and initial message passing. During rollout, the same previous `edge_state` is passed as both `physical_edge_state` and `info_edge_state`, then fused with the new action. Thus future physical and information latents are not independently propagated after initialization.
- Task dynamics remain a single/global latent updated from mean node and edge states. There is no explicit per-task graph, queue conservation mechanism, or typed task-flow transition in this core class.
- These choices are computationally simple and compatible with fixed-shape logs, but they can erase sparse local interventions and stage-specific task effects. A future architecture could preserve separate physical and information latent streams across rollout and connect task tokens to the exact communication/compute entities involved.
- The optional `stgcn_full` path constructs node and line-graph-style edge adjacencies, while the standard message-passing path uses endpoint-conditioned edge updates. These are alternative history mechanisms, not necessarily simultaneous components of the best model.

### Dataset Assembly and Training Protocol

- The world-model dataset is assembled by strict alignment of state and action samples on seed and, when available, sample ID. It hard-fails on inconsistent history/horizon/entity dimensions or an all-zero future-action tensor.
- Link activity is operationally defined as `y_link_rate > 1e-6`; it is not an independently logged semantic label. Therefore activity F1 measures consistency with the chosen rate threshold, and conclusions about communication events should preserve this definition.
- State tensors and action tensors are stored separately before assembly. This is good for alignment audits, but counterfactual validity still depends on whether action channels faithfully represent the simulator scheduler's causal intervention, not merely whether arrays line up.
- The v8 runner exposes explicit seed lists and validates split disjointness, but its CLI/default fallback still uses historical seeds 0-7/8/9. The current README rule requiring explicit seed lists is necessary because code defaults remain backward-compatible.
- Checkpoint selection is validation-driven and can target active-rate RMSE or constrained composite metrics. Activity thresholds are chosen on validation and then applied to test, which is protocol-correct when the configuration is otherwise frozen.
- The default/smoke CLI (1 epoch, 64/32/32 samples, hidden 32) is not a research configuration. Reproducible scientific claims must point to the exact artifact summary/command rather than quote runner defaults.

### Artifact and Literature Locations

- The experiment archive is very large and contains many v10/v11 branches. Its existence is valuable for negative-result provenance, but directory names alone cannot establish the current best configuration.
- A dedicated `代码/artifacts/literature/` directory exists even though the tracked `文档/文献` directory was absent from the initial inventory. Historical group-meeting archives also contain PDFs, including digital-twin/generative-AI and low-altitude wireless-network papers.
- Formal claims should be traced to structured summaries, commands, manifests, and locked result directories rather than inferred from the latest-looking experiment folder.

### Local Literature Corpus and Citation Coverage

- The focused local PDF set covers adaptive graph learning (Graph WaveNet, AGCRN), continuous-time temporal graphs (TGN), decoupled dynamic STGNNs (D2STGNN), and uncertainty for sparse traffic forecasting. These are directly relevant to dynamic topology, periodicity, sparse events, and uncertainty, but not all are cited in the current paper draft.
- The paper draft cites strong foundational buckets: MEC/VEC/UAV offloading, GNNs for wireless resource management, STGNN surveys, wireless digital twins, multi-task learning, selective/conformal prediction, offline support constraints, decision-focused learning, POMDP/world models, MPC, and AirFogSim.
- The current bibliography underrepresents the closest 2023-2026 domain competitors already found in the repository's novelty audit, including FlexEdge, RADiT, adaptive DT for UAV-assisted ISCC, and recent action-conditioned/agentic wireless world-model preprints. This weakens the novelty argument even if the method itself remains defensible.
- The literature should be organized by the problem it constrains, not by borrowing modules: dynamic relational forecasting; causal/action-conditioned models; wireless DT/world models; offline counterfactual validity; sparse event/magnitude modeling; and decision evaluation under uncertainty.

### Focused STGNN/Sparsity Literature Reading

- Graph WaveNet/AGCRN challenge the assumption that a fixed physical adjacency equals predictive dependency. AGCRN specifically learns node-specific parameters and a data-adaptive graph, which is relevant because PI-JWM currently shares update parameters and uses fixed indexed edges.
- TGN treats dynamic graphs as timed interaction events with memory. PI-JWM's sparse link activations are closer to an event process than a dense snapshot signal, but the earlier naive event-memory concat polluted rate prediction. A typed event memory routed by event family/stage is more faithful than a global memory feature.
- D2STGNN's dynamic-versus-inherent decomposition supports separating propagated interaction effects from local baseline dynamics. This is conceptually richer than only splitting activity versus positive magnitude.
- STZINB-GNN provides a valid warning about excess zeros and uncertainty, but its negative-binomial likelihood is for sparse count demand. PI-JWM rate is continuous and heavy-tailed, so the transferable idea is a two-part/distributional model, not the literal ZINB distribution.
- The v9 literature plan correctly concluded that standard STGCN, naive MoE, and module stacking are not automatic improvements. Its strongest durable lesson is to match an architectural mechanism to a diagnosed failure mode and test one mechanism at a time.
- The local literature mostly targets observational forecasting. It does not by itself establish reliable counterfactual prediction under candidate actions; that gap needs causal/offline model-based evaluation literature, not more STGNN variants alone.

### External Verification Status

- The exact arXiv IDs `2511.02748` and `2606.11618` cited in the local novelty audit returned no Atom entries on 2026-07-21. Until independently verified, their titles and claimed overlap must not be used as evidence in the paper.
- OpenAlex is reachable, but broad full-text search is noisy and PowerShell's legacy JSON parser fails on some case-variant keys in abstract inverted indexes. Subsequent queries will request metadata-only fields and verify promising records at first-party DOI/arXiv pages.

### Verified Closest Domain Work

- OpenAlex metadata verifies FlexEdge (TVT 2023, DOI `10.1109/TVT.2023.3262261`), RADiT (JSAC 2023, DOI `10.1109/JSAC.2023.3310048`), Adaptive Digital Twin for UAV-Assisted ISCC Networks (TGCN 2023, DOI `10.1109/TGCN.2023.3298039`), and UAV-Assisted Task Offloading in Vehicular Edge Computing Networks (TMC 2023, DOI `10.1109/TMC.2023.3259394`).
- These works occupy much of the broad application territory: UAV-aided vehicular edge computing, digital twins, trajectory/resource/offloading decisions, energy and delay objectives, and DRL/MAPPO-style optimization. PI-JWM cannot claim novelty at that level.
- A 2026 indexed paper, `Digital Twin-Assisted Large AI Task-Aware Edge Offloading and Resource Allocation for Low-Altitude Wireless Sensor Networks` (DOI `10.1109/JSAS.2026.3679846`), indicates that low-altitude DT-assisted task-aware offloading is still an active and increasingly close line.
- The defensible distinction is whether prior systems learn a reusable, action-conditioned multi-step joint state transition over physical geometry, link activity/rate, task state, and resources, with explicit counterfactual-action and deployability audits. That exact distinction still needs paper-level comparison, not title-level assertion.
- Semantic Scholar returned no usable payload for the exact 2026 agentic titles or broad world-model queries, reinforcing rather than resolving the arXiv non-match. Those two local-audit records remain excluded from verified evidence.

### Offline World-Model Literature Implications

- MOPO penalizes imagined reward by learned-dynamics uncertainty; MOReL constructs a pessimistic MDP; COMBO regularizes values on model-generated out-of-support state-action tuples; RAMBO/ARMOR optimize against adversarial plausible models. Their common lesson is that a learned rollout model is easiest to exploit precisely where logged support is weak.
- PI-JWM currently applies defer/Pareto/LCB logic mainly at the candidate selector layer. A stronger scientific design would also estimate action-conditioned rollout support and epistemic uncertainty inside the world model, per edge-step and per prediction head.
- The right question is not only "which candidate has lower predicted RMSE?" but "for which parts of this candidate trajectory is the learned transition trustworthy, and how does uncertainty compound across rollout steps?"
- DreamerV3 and TD-MPC2 show the value of integrated latent dynamics/planning at scale, but copying their control machinery would not solve PI-JWM's central issue: logged wireless actions are sparse, structured, and support-constrained, and the current dataset does not permit unconstrained imagination.
- A promising bridge is conservative counterfactual evaluation rather than full offline RL: learn paired local transition deltas with support scores, propagate intervals through the joint state model, and only compare candidate sequences where the relevant edge-stage interventions are identifiable.

### v11 Causal Candidate Protocol and Phase Selector

- `causal_policy_v1` deliberately removes absolute task IDs. A candidate freezes a family, coverage, rank, scale, and optional return-route mode; each rollout step resolves intent against the currently legal tasks. This fixes a real temporal-executability flaw in precomputed task-ID actions.
- Supported families are default, RB count/scale, offload target, mixed offload-RB, CPU scale, and return route. Count scaling uses deterministic capacity-constrained projection, making action realization auditable.
- The phase selector is a train-time lookup table over exact episode local step (390 phases) and candidate identity. For each cell it stores mean raw-SSE benefit, standard deviation, positive direction rate, and support count across train outcomes.
- Calibration chooses only the LCB multiplier, positive-rate threshold, minimum mean benefit, and minimum count under precision/negative-selection gates. Online validation uses phase, candidate legality, and observable Pareto proxies; it does not use validation outcomes or `action_applied` labels.
- This is deployable under the repeated 390-step experiment protocol, but it is not a generally learned state-conditioned policy. It assumes that local step is a stable surrogate for latent task stage and that candidate identities have repeatable effects across seeds.
- The strong 10/10 validation result therefore supports a substantive discovery: the data has a highly repeatable phase structure. It does not yet establish robustness to changed episode length, traffic schedule, task-arrival process, topology, or scenario.
- The right next use of phase is as an auxiliary/weak supervisory signal for explicit task-stage inference, not as the permanent primary state representation.

### v11 Label Semantics and Interaction Features

- Candidate `active_sse` labels compare each candidate-conditioned PI-JWM rate prediction against the single logged factual future. A positive benefit means the candidate makes the frozen model reconstruct the observed future more accurately than the default action.
- This is not the same as executing the candidate in AirFogSim and observing lower delay, energy, or higher task utility. Therefore v11's 207.5399 active-rate RMSE is an improvement in autonomous action-condition reconstruction/predictive consistency, not direct evidence of better system control.
- The physical-benefit bridge separately adds paired simulator task/energy outcomes and deployable proxies, but the strongest phase-selector metric is still prediction-error benefit. Paper language must not collapse these objectives.
- Base candidate features summarize actions, PI-JWM predictions/deltas, task forecasts, action family, selected-edge current statistics, and global history context. The schema-v6 extension adds up to 72 modified edge-step tokens with 25 fields and a 234-dimensional auditable pooling contract.
- Each token includes step, default action, six action deltas, five current link features, default predicted activity/rate, and predicted response deltas. It does not include explicit source/destination node state, task identity/stage binding, or a persistent edge embedding.
- This helps explain the audit result "opportunity detectable, within-sample ranking weak": global context can identify hard phases, but tokens may not contain enough causal identity to distinguish which modified edge/task path will truly matter.
- The project should distinguish three objectives explicitly: factual state forecasting under logged actions; inverse action reconstruction from observed futures; and counterfactual decision optimization under task/energy utility. They share machinery but require different labels and evaluation protocols.

### Frozen Artifact Entry Points

- The formal phase-selector directory contains a summary, freeze payload, per-seed metrics, full validation decision trace, train statistics, reproduction command, source SHA, and SHA-256 manifest. It is the correct evidence source for v11's latest validation result.
- The v10 action-aligned world-model package contains frozen checkpoints, two training summaries, action ablations, metrics, model comparisons, a freeze manifest, and checksums. It is the correct source for the true-future world-model claim.

### Frozen Evidence Corrections and Limits

- V10's main model is dual-graph, cross-attention, recurrent latent transition, and hurdle-style rate output on a 23,400-sample, 60-seed, horizon-3 dataset with 58 nodes and 314 candidate edges.
- The test has only 396 active edge-steps out of about 734,760 edge-steps for seeds 18-19. A deterministic `rb_task_count > 0` rule achieves activity F1 1.0, so V10's 0.902 activity F1 is largely enabled by the realized future action condition and should not be presented as autonomous event forecasting.
- Zeroing future actions increases active-rate RMSE from 100.14 to 502.55; zeroing historical actions slightly improves it to 97.71. The model is strongly future-action dependent, while historical action dynamics add little in this ablation.
- The V10 train split includes seeds 50-59, which later serve as v11 selector validation. Thus v11 validation is held out from selector fitting but not from world-model fitting. It measures selector generalization on a frozen in-domain dynamics model, not end-to-end unseen-seed generalization.
- The formal phase selector is `diagnostic_only`, not frozen, with external holdout locked. It defers on 95.35% of valid validation samples and executes only 65 interventions.
- Despite the method name, the calibrated formal configuration uses `z_value=0.0`; selection is based on mean benefit plus positive-rate, mean-benefit, support-count, legality, and Pareto gates. No variance penalty is active in the winning rule, so calling the frozen rule an LCB is technically imprecise.
- Autonomous validation activity F1 remains only 0.02768. The 26.18 active-rate RMSE improvement is real under its metric, but the overall autonomous link-event reconstruction problem is far from solved.
- The sharp jump from the 16-seed control (216.72) to the 40-seed V10 main model (100.14) combines more data, broader seed coverage, and action alignment. It should not be attributed to architecture alone.

### Physical Benefit Bridge Design

- The bridge correctly recognizes that historical immediate-intervention simulator results cannot supervise selector-timed step-1/2 interventions. It requires exact sample-time alignment, shared default step 0, action applicability/effect audits, paired default controls, and split-isolated training/calibration.
- Its features are public action descriptors plus current task/link/resource and PI-JWM forecast context; true simulator task/energy outcomes remain supervision/audit only. The design uses seed-group cross-fitting and conformal intervals.
- Physical benefit enters the selector primarily as a conservative Pareto feasibility gate (task LCB, energy UCB), while active-rate SSE regret remains the ranking objective. Thus even a successful bridge does not turn the selector into a task-delay/energy optimizer.
- Candidate families without demonstrated semantic correspondence between AirFogSim and selector actions are correctly excluded rather than force-mapped.
- The bridge design is one of the strongest parts of the project methodologically: it treats simulator semantics, temporal intervention timing, leakage, and uncertainty as first-class contracts rather than post-hoc metrics.

### Physical Bridge Results

- Formal horizon-10 simulator data covers 40 train seeds (120 decision groups, 809 candidates, 8,090 step rows) and 6 calibration seeds (18 groups, 117 candidates, 1,170 step rows), with complete alignment and quality audits.
- The longer, selector-timed persistent/decayed protocol raises nontrivial coverage to 90/120 train groups and 15/18 calibration groups, substantially better than the earlier 7/15 immediate-intervention diagnosis.
- Only the task model passes: OOF task MAE 0.16023 versus 0.16310 baseline, and calibration task MAE 0.07914 versus 0.08176. The margin is small, so the result demonstrates weak identifiability rather than a highly accurate physical surrogate.
- The energy model fails badly: OOF MAE 1.429 versus 0.862 baseline and calibration MAE 2.484 versus 2.252. Its conformal radius is 7.927, so predicted physical energy is correctly excluded from online features and retained as audit-only.
- The task-only bridge augments caches with task mean/std/LCB/UCB; online energy safety still relies on a deployable proxy. No actual future outcome appears in selector features.
- This outcome narrows the research claim: task effects are weakly learnable from the current descriptors, while UAV energy response is not. A task-energy co-optimization claim is premature.

### Handoff Audit

- The 2026-07-21 handoff accurately preserves framework identity, data locks, selector root-cause evidence, failed routes, code entry points, and open research questions.
- Its literature section still presents the two unresolved 2026 arXiv titles as established prior work. Given the failed arXiv/OpenAlex/Semantic Scholar cross-check, those entries should be marked unverified before reuse in advisor-facing text.
- Per-seed validation confirms all ten selector seeds improve versus their ranked defaults, but oracle gaps remain large; seed 55 is the hardest selected result (257.91 versus 271.30 default and 95.67 oracle).

### Phase Decision-Trace Structure

- Of 65 executions, 58 select `benefit_residual__shrink50__k16`; the remaining seven are q50 RB-repair variants. The improvement is therefore dominated by one conservative RB shrink action, not broad candidate-family ranking.
- Sixty executions occur at only six phases: 71, 72, 111, 112, 151, and 152, each repeated across all ten validation seeds. Only five executions occur at four later phases (323, 329, 366, 369).
- This pattern strongly suggests a periodic scheduler/task-arrival structure at roughly 40-step intervals. The selector has discovered a time-indexed repair schedule.
- Four of the 65 executions have negative realized SSE benefit; three occur at the repeated phase 71/72 and one at phase 329. The overall positive precision remains high because the repeated intervention is stable across most seeds.
- A minimal next scientific test is not another deep ranker: perturb episode start, task-arrival phase, or traffic schedule while holding marginal load similar, then test whether the same phase table fails and whether state/event-stage features recover the benefit.

### Phase-to-State Analysis

- The base dataset has history 8, horizon 3, 58 nodes, 314 edges, 7 node features, 5 link features, 9 task features, and 6 action channels. All 60 seeds contain exactly 390 contiguous samples.
- The six repeated execution phases coincide with extreme future activity/RB bursts. Phases 71/72, 111/112, and 151/152 rank in roughly the 98th-100th percentiles for future RB-task count and true active link-steps across validation seeds.
- Phase 71/72 averages 2.6-2.8 future active steps and 33.4-43.4 RB total; 111/112 averages 3.6-3.8 active steps and 31.5-39 RB; 151/152 averages 3.2-3.3 active steps and 22.5-30 RB. These are not arbitrary clock positions; they mark scheduled communication bursts.
- Observable task state also changes systematically: total tasks rise from about 12.5 at phase 71 to 32.5 at 111 and 60.5 at 151, while mean deadline/priority shifts. Phase is aliasing a deterministic event schedule and cumulative task lifecycle.
- The main selector gain can therefore be reframed as correcting ranked-allocation RB magnitude around predictable event onsets. This directly connects the v11 `rb_total` diagnosis to an event-timing problem.
- A more general model should predict a hazard or time-to-next communication event from task queue/stage and scheduler state, then estimate the required RB mass conditionally. It should be invariant to absolute episode start and tested under shifted arrival schedules.

### Multi-Perspective Synthesis

1. **Wireless-systems reviewer** asks whether PI-JWM currently solves joint delay-energy control. Evidence says no: the world model predicts joint state, v11 reconstructs useful action conditions, and the physical bridge only weakly predicts task delta while energy is audit-only. The control problem remains an ambition.
2. **Dynamic-graph/world-model reviewer** asks whether physical and information dynamics stay distinct and topology-aware. v8 has real endpoint message passing, but uses fixed edges, collapses both future edge streams into one latent, and globally pools task state. The current architecture captures correlation better than typed causal flow.
3. **Causal/offline-decision reviewer** asks whether counterfactual candidate effects are identified. Main selector labels are factual prediction-error deltas, and logged-action support is sparse. Simulator-paired task/energy labels improve semantics but remain limited. World-model support/uncertainty must be estimated before planning outside logged actions.
4. **Statistician/evaluation reviewer** asks what actually generalizes. Seed-held-out selector validation is rigorous relative to selector fitting, but the world model has trained on those seeds; exact phase exploits a fixed 390-step schedule; matched test is consumed and external is locked. End-to-end scenario transfer is unproven.
5. **Paper reviewer/advisor** asks what contribution survives these caveats. The defensible core is a domain-specific, action-conditioned physical-information state model, sparse event/magnitude evaluation, and an unusually careful simulator-aligned diagnostic protocol. A final autonomous controller, task-energy optimizer, generic foundation model, and broad novelty claim do not yet survive.

### Independent Research Directions

#### Route A: Event-Aligned Joint World Model

- Preserve separate physical and information latent streams across rollout, add explicit task tokens/stages and task-edge/node incidence, and model activity as a next-event hazard with conditional rate magnitude.
- Replace absolute phase with learned time-to-next-event/task-stage features; phase remains an ablation teacher only.
- Minimal falsification: train a small hazard + conditional RB-mass model on existing train seeds and evaluate on validation after cyclically shifting phase indices. It must beat ranked allocation and the phase table under the shift, not only on the original clock.

#### Route B: Support-Aware Counterfactual Delta Model

- Generate paired simulator transitions for a small set of persistent, causally executable local repairs and learn `delta next-state/task/energy` relative to default, with per edge-step support scores and epistemic intervals.
- Use pessimistic selection inspired by offline model-based RL; reject a candidate when any critical intervention lies outside support.
- Minimal falsification: on train/calibration only, test whether predicted delta sign and interval coverage generalize by seed and family. Stop unless sign precision, coverage, and actual task/energy direction all beat simple stage-family baselines.

#### Route C: Structured Resource Projection Instead of Generic Ranking

- Treat event support and resource magnitude separately. Predict active task-edge demand, then solve a constrained RB/CPU projection satisfying capacity, count-total coupling, and task priority/deadline constraints.
- The current decision trace suggests one repeated failure is RB over-allocation near burst onset; a structured shrink/redistribution rule may be more data-efficient than ranking 32 opaque templates.
- Minimal falsification: compare a queue/channel-conditioned analytic projection against `shrink50_k16` on the same validation cache, preselecting all parameters on calibration. Require gains across shifted phases and multiple action families.

#### Route D: Scenario-Transfer Track

- Do not call seed diversity foundation pretraining. Build scenario diversity across road maps, mobility/load regimes, task mixes, channel parameters, UAV missions, and episode timing.
- Pretrain masked event/state reconstruction and action-conditioned transition modules, then test leave-one-scenario-out few-shot adaptation.
- Minimal falsification: two genuinely distinct simulator configurations are enough to test whether pretrained representations reduce target-scenario sample complexity versus training from scratch.

### Recommended Research Split

- A forecasting paper should prioritize typed joint dynamics, calibrated sparse event/rate prediction, scenario transfer, and counterfactual validity.
- A decision paper should prioritize executable candidate sequences, simulator-paired task/energy outcomes, conservative support-aware selection, and online utility.
- Combining both is possible later, but the current evidence is clearer if PI-JWM's main contribution remains the world model and v11 remains a diagnostic interface.

## 2026-07-22 Literature and Workspace Audit

- The user has created an empty Zotero collection named `PIJWM` and wants Zotero to become the authoritative literature manager.
- The requested acquisition policy is recent work from approximately 2023--2026 in top journals, with exceptions for foundational or still-unreplaced ideas.
- Root cleanup is potentially destructive and “outdated” is not yet a file-level criterion; inventory and a candidate manifest must precede deletion.
- Root currently includes `.git`, `.agents`, `.codex`, repository configuration, `README.md`, `AGENTS.md`, `pyproject.toml`, three planning/audit Markdown files, `新对话接续说明.md`, and `本地计划表.md`, alongside `代码/` and `文档/`.
- Zotero Desktop 7 is running from `C:\Program Files\Zotero\zotero.exe`; the standard `C:\Users\Lenovo\Zotero\zotero.sqlite` path was not found in the first probe, so the configured data directory still needs to be resolved from the active profile.
- The repository worktree was already dirty before cleanup: user changes exist in the two research TeX documents/PDF, `研究问题与系统边界.md`, and `本地计划表.md`; three engineering-governance documents are already deleted; `新对话接续说明.md` is untracked. These must be preserved and not conflated with cleanup changes.
- The workspace contains 46 PDFs. The research corpus is split across `文档/文献/`, `文档/研究进展/papers/ton_references/`, `代码/artifacts/literature/`, and historical meeting archives; several obvious byte-level or title-level duplicates are likely, including the v9 graph papers and low-altitude-network papers.
- Non-literature PDFs also appear in the count (generated PI-JWM drafts, IEEE templates, meeting reports), so file extension alone cannot determine Zotero eligibility.
- The active Zotero profile is `t1pynyyu.default`; it explicitly enables the Zotero local API and uses a custom data directory at `D:\禹尧珅\人工智能知识库\科研` rather than the default `C:\Users\Lenovo\Zotero` database.
- Initial read-only requests to `127.0.0.1:23119` were closed by the peer even though the preference says the local API is enabled. The next diagnostic is to inspect Zotero's actual listening sockets and use the bound address/protocol.
- Zotero preferences contain private service credentials. Future diagnostics must query only named non-secret settings and must not archive or reproduce the whole preferences file.
- Zotero local API access is confirmed with `curl --noproxy '*'`: Zotero version 9.0.5 reports 18 collections and 94 top-level items. The `PIJWM` collection key is `MZ9JQ2I6` and contains zero items.
- All 94 top-level Zotero items have child items/attachments, 87 have DOI metadata, and seven lack DOI metadata. Several PI-JWM-relevant local PDFs are already represented in Zotero under older collections, so they should be added to `PIJWM` rather than imported as duplicates.
- Zotero already contains seven exact duplicate DOI groups, all in non-PIJWM topic collections. They should be reported separately; merging them is outside the immediate PIJWM scope unless the user approves a whole-library deduplication.
- Six clearly PI-JWM-relevant items share an unlisted/stale collection key `5AY4BRZU`: dependency-aware UAV offloading, ISCCC/UAV swarms, low-altitude network control, RoboScape, world-model survey, and digital-twin-to-world-model preprint. Their item records are intact and reusable.
- Of 46 workspace PDFs, 45 have unique SHA-256 hashes. The only exact duplicate is Zhuang et al.'s 2022 sparse traffic uncertainty paper, stored once under `代码/artifacts/literature/...` and once under `文档/文献/...`.
- `文档/` currently has four main areas: `文献` (17 files, 15 PDFs), `项目说明` (18 files, 16 Markdown), `研究进展` (50 files, 19 PDFs), and `组会` (1361 files, including 115 Markdown and 37 PPTX). Most clutter is therefore concentrated in the historical meeting archive and generated preview assets, not the active project root.
- Root has seven Markdown files. `README.md` and `AGENTS.md` are active repository guidance; `本地计划表.md` is the current authoritative plan under existing repository rules; `task_plan.md`, `findings.md`, `progress.md`, and `新对话接续说明.md` are archival/handoff candidates after this consolidation finishes.
- `文档/研究进展` contains disposable LaTeX build byproducts (`.aux`, `.log`, `.out`, a zero-byte `.synctex(busy)`, and `__pycache__`) alongside source TeX/PDF. These are high-confidence cleanup candidates, subject to the final review gate.
- Zotero's official Web API documentation (updated 2026-07-07) states that the desktop local API exposes the same read endpoints at `http://localhost:23119/api/`, requires no authentication, and is described as read-only. Therefore it is suitable for inventory and verification but not the chosen write path for collection creation/import.
- Zotero mutations should use supported desktop UI/import mechanisms (or an explicitly authorized Zotero Web API key, which is unnecessary here). Direct SQLite writes remain prohibited.
- The current ToN draft already names 24 core/support references. Recent top-venue items include Shen et al. (TWC 2023), ENGNN (TWC 2024), two 2024 graph/time-series surveys (TKDE/TPAMI), Tao et al. (IEEE Wireless Communications 2024), Mandi et al. (JAIR 2024), and AirFogSim (TMC 2025). These form the first deterministic import set before adding new search results.
- The v8 literature map also contains later supporting papers not fully represented in the main bibliography: knowledge distillation (IJCV 2021), MoE survey (TKDE 2025), conformal prediction review (Bernoulli 2023), and the graph/time-series method set in the v9 artifact corpus.
- The initial 2023--2026 OpenAlex discovery returned relevant candidates in five themes, including `6G Digital Twin Networks: From Theory to Practice`, `Digital Twins for 5G Networks`, `Colosseum as a Digital Twin`, `Graph Neural Network-Based Continual Learning for Resource Allocation in Dynamic Wireless Environments`, several recent UAV-MEC joint offloading/resource papers, and `RADiT`. These are discovery candidates only until DOI/venue/full-text verification.
- Broad world-model and real-dataset keyword queries were noisy. They need narrower domain-constrained queries and first-party verification rather than automatic inclusion.
- Exact metadata verification identified strong recent candidates: 6G Digital Twin Networks (IEEE Communications Magazine 2023, DOI `10.1109/MCOM.001.2200830`), Colosseum as a Digital Twin (TMC 2024, `10.1109/TMC.2024.3359596`), WMMSE-unrolled GNN resource allocation (IoT-J 2024, `10.1109/JIOT.2024.3368516`), continual GNN resource allocation (TVT 2025, `10.1109/TVT.2025.3585146`), and RADiT (JSAC 2023, `10.1109/JSAC.2023.3310048`).
- Strong recent UAV/MEC candidates include two TMC 2024 joint task/resource papers (`10.1109/TMC.2024.3350886`, `10.1109/TMC.2024.3350078`), a TMC 2023 air-ground energy/latency paper (`10.1109/TMC.2023.3346431`), and dependency-task scheduling in JSAC 2023 (`10.1109/JSAC.2022.3233532`).
- For strict dual-graph theory, search results do not reveal one canonical recent paper that directly defines PI-JWM's physical-entity graph plus task-dependency graph coupling. The defensible evidence chain is composite: attributed graph networks for typed nodes/edges, wireless edge-update GNNs, dynamic graph representation, and DAG-dependent task scheduling/offloading. The project should not falsely claim that a single prior paper proves this exact construction.
- Additional high-value graph/task references discovered are RouteNet-Fermi (IEEE/ACM ToN 2023, `10.1109/TNET.2023.3269983`), GASTO (TNSM 2023, `10.1109/TNSM.2023.3250395`), dependent-task deadline scheduling (JSAC 2023), real-time dependent/parallel offloading (TPDS 2024, `10.1109/TPDS.2023.3349177`), and DAG scheduling in IoV (TMC 2025, `10.1109/TMC.2025.3531887`).
- The closest published wireless/world-model bridge papers discovered are AGI-Native Wireless Systems (Proceedings of the IEEE 2025, `10.1109/JPROC.2025.3526887`), Edge General Intelligence Through World Models (TCCN 2026, `10.1109/TCCN.2026.3658762`), and model-based DRL for wireless channel access (IoT-J 2023, `10.1109/JIOT.2023.3325575`). They are closer conceptually than generic robotics world-model work but still do not implement PI-JWM's strict physical-task dual graph.
- Two JSAC 2023 papers are especially relevant to the decision bridge: digital-twin-driven heterogeneous task/resource scheduling (`10.1109/JSAC.2023.3310066`) and GNN-based service self-healing in 6G edge networks (`10.1109/JSAC.2023.3310063`).
- No single realistic public dataset appears to contain mobility, per-link channel/traffic, task DAG/lifecycle, RB/CPU actions, and energy outcomes together. The evidence supports a hybrid data strategy rather than replacing AirFogSim wholesale.
- High-value data/testbed references are Colosseum/TMC 2024, `Where Are the (Cellular) Data?` (ACM Computing Surveys 2023), OpenRAN Gym (WCNC 2022), DeepMIMO (foundational 2019), DeepSense 6G-related real multimodal blockage/beam data, and AERPAW full-stack drone measurements. These mainly calibrate or externally validate the physical/communication side; they do not supervise the full task-resource transition.
- The first lawful OA acquisition batch contained 27 verified DOI records. Twelve PDFs were downloaded successfully and fifteen remain metadata-only after the default OA resolution chain; Sci-Hub was explicitly disabled and institutional mode was not enabled.
- Downloaded additions include the TPAMI graph/time-series survey, RouteNet-Fermi, 6G Digital Twin Networks, both Colosseum papers, AGI-Native Wireless Systems, JSAC GNN service self-healing, OpenRAN Gym, DeepMIMO, the O-RAN survey, and two real-world blockage prediction papers.
- The fifteen unresolved items are concentrated in IEEE task-offloading/resource-allocation papers and a few data/survey papers. They require author-repository/OA-location checks; inability to find a lawful full text will result in a complete metadata record with `PIJWM/全文/仅元数据`, not an illicit fallback.
- OpenAlex OA-location verification reduced the unresolved set: the Computer Networks digital-twin survey and `Where Are the (Cellular) Data?` were downloaded from HAL and validated by `%PDF-` magic bytes. Two IEEE records are marked public/hybrid and load successfully in the user's normal Chrome PDF viewer; the remaining eleven are currently closed/metadata-only.

## 2026-07-22 Theoretical Definition Grounding

- The current 60-seed dataset uses 8 history steps and 3 future steps with 58 nodes and 314 directed candidate edges.
- Node fields are `x, y, z, speed, acceleration, cpu, storage`; node state therefore mixes physical motion and resource state rather than being purely physical.
- Information-edge fields are `distance, rate_sum, csi_mean, active_task_count, allocated_rb_count`; physical-edge features are separately derived from endpoint position and speed.
- Task state is a 9-dimensional global aggregate, not per-task tokens: total count/size/CPU, mean deadline/priority, and counts in offload/compute/return/finished stages.
- Edge action is a 6-dimensional aggregate: offload count, RB task count/total, CPU task count/total, and return count.
- Current targets cover node state, link activity, link rate, and aggregate task state. They do not directly contain per-task arrival/completion timestamps or actual energy.
- Consequently, per-task mean/P95 delay, strict deadline violations, and energy per completed task are desired system objectives but are not yet direct world-model targets. They require raw-log audit and probably dataset/schema extension.
- Recommended final objective candidate is Pareto minimization of `[-completion ratio, completion delay, energy per completed task]` under legality/resource/deadline/energy constraints. This avoids the trivial no-service low-energy solution and avoids post-hoc scalar weights.

## 2026-07-24 RRM/World-Model Review Intake

- The folder `基于信道图谱的RRM` contains two review PDFs, one PPTX, and one Markdown speaker-note file.
- `ICT_World_Models_Survey_Communication_Networks_and_MAC_Layer.pdf` is a 37-page English structured scoping review. `pypdf` extracted about 101,631 characters successfully. Its stated central conclusion is that a communication-network world model should learn a compact internal world that predicts task consequences under actions, combining explicit protocol state, latent wireless/traffic beliefs, graph topology, and uncertainty.
- Direct `pypdf` access to `ICT世界模型文献调研综述_通信网络与MAC层.pdf` was denied by the OS after the Unicode path was passed through the process environment. This is an access/encoding issue, not a content finding; next attempt will copy the file with PowerShell to an ASCII temporary path before extraction.

### English review: first-pass findings

- The review defines a strict world model as a decision-oriented internal environment model with a state/belief representation, action-conditioned dynamics, outcome heads, and a planning interface. A one-step predictor, CSI predictor, digital twin, foundation model, or model-free policy is not automatically a world model.
- Its central criterion is counterfactual action consequence: the model must answer what follows after candidate action $a$, not only what traffic/CSI/load will be next under the historical policy.
- The recommended communication/MAC representation is hybrid: explicit protocol state, wireless and traffic latent beliefs, a topology/interaction graph, and uncertainty. It should not attempt to reconstruct every IQ/CSI coordinate unless the task requires it.
- The recommended MAC route is: define objective and hard constraints; construct a timestamped/masked hybrid state; train action-conditioned dynamics with multitask outcome heads; plan with MPC, search, differentiable imagination, or a solver; then enforce an independent safety filter, baseline fallback, OOD/drift monitoring, and closed-loop KPI evaluation.
- The review stresses that model loss and control quality are different evaluation layers. Low one-step MSE does not prove useful action ranking; rollout error, action sensitivity, constraint violations, calibration, tail reliability, online latency, and closed-loop KPI must also be measured.
- The review treats counterfactual coverage as a central data problem: logs only label executed actions, so simulation, protocol rules, multi-policy data, and matched-budget evaluation are needed to avoid unsupported action extrapolation.
- The accompanying speaker notes interpret the communication actions broadly: access/retreat, user/link selection, RB/time-slot allocation, power, MCS, retransmission, and cross-layer resource actions. They also distinguish the world model from the planner/strategy layer and recommend a safety shell independent of the learned model.

### English review: representation, planning, and training details

- The review frames the problem as partially observed control: an inference/belief module maps observation history and past actions to an internal state, then an action-conditioned transition predicts the next state. Outcome heads should expose observations, task/KPI values, reward, constraints/risk, and uncertainty.
- A useful representation must preserve five things: inferable current state, memory of history, action-affected degrees of freedom, future uncertainty, and task/constraint-relevant value. It should make equivalent states predict similar futures and make distinct actions produce distinguishable consequences.
- The review recommends a hybrid rather than a universal representation: explicit queues/protocol variables and rules for auditability, latent wireless/traffic state for compression, graphs for topology/conflict, and uncertainty for missing data and risk. The network analogue should be a radio field plus moving entities and a dynamic interaction/conflict graph.
- Important failure modes are state aliasing, latent drift, graph oversmoothing on unseen topology, causal shortcuts, and planner exploitation of optimistic model errors. One-step MSE is explicitly insufficient; multistep calibration, action sensitivity, OOD detection, conservative estimates, and closed-loop tests are required.
- Controllable variables include power, beams, scheduling, routing, caching, slicing, and offloading. Exogenous variables include propagation, mobility, weather, demand, failures, and other agents. The model should not learn accidental correlations as if external events were controllable.
- Time scales should be separated: fast PHY/MAC service/conflict/HARQ, medium queue/mobility, and slow traffic/configuration/radio-field evolution. A hierarchical or event-driven design is preferred over one fixed step for all phenomena.
- The planner is not the world model. CEM/MPPI fits continuous resources, MCTS fits small discrete choices, actor-critic imagination can reduce online cost, and symbolic optimization is suited to combinatorial hard constraints. A hybrid neural-dynamics plus solver plus safety-filter design is recommended.
- Data must cover actions, not just observations. Single-policy logs create action-state confounding and unsupported counterfactuals. The review recommends multi-policy logs, safe simulation exploration, rare-action sampling, behavior constraints, uncertainty penalties, and protocol/physical inductive biases.
- A practical training objective may combine observation/outcome, KPI/reward, constraint, dynamics consistency, predictive representation, KL, inverse-dynamics, and calibration losses. Loss weights should be judged by closed-loop and calibration performance rather than reconstruction alone.

### English review: communication-network implications

- The review organizes the wireless world into four layers: geometry (3D structure, blockers, location), propagation state (multipath, delay, angle, fading), measurements (IQ/CSI/RSRP/RSRQ/SINR), and service outcomes (rate, loss, delay, energy). CSI/channel state is therefore an intermediate prediction target, not the complete RRM objective.
- Resource management needs joint prediction of physical and service outcomes. A slow radio-field/traffic model and a fast local MAC model should exchange scenarios, priors, and uncertainty; millisecond MAC control cannot directly use the same expensive model used for hourly planning.
- Task-centric communication extends the objective beyond throughput: the value of a link may be the improvement in remote belief, control loss, AoI, deadline satisfaction, or sensing quality. Confidence must be calibrated, and prolonged missingness should trigger measurement or conservative fallback.
- The review distinguishes a network digital twin (asset/configuration/data/lifecycle container), a world model (learned belief and action-conditioned dynamics), an optimizer/planner, and an LLM interface. These layers should not be conflated.
- The communication evidence is an inverted pyramid: many conceptual and simulation studies, few common benchmarks and replications, and very few live deployments. Any PI-JWM claim should state scenario, action support, time scale, data source, and evidence level.
- Distinctive constraints are configuration-dependent semantics, action-feedback selection bias, multi-agent nonstationarity, hard PHY/protocol limits, rare tail events, and microsecond/millisecond latency. These are more important for project redesign than adopting a fashionable architecture.

### English review: MAC formulation and evaluation

- MAC state should include queues, arrival rate, channel/service rate, AoI, deadlines, HARQ, conflict relations, and energy. Actions include access/backoff, user/link selection, RB/slot assignment, power, MCS, retransmission, and anchor activation.
- The review gives an explicit queue conservation law: $q_{i,t+1}=[q_{i,t}-\mu_i(z_t^{radio},a_t)]^++\lambda_{i,t}$. The service rate depends on wireless state and action; queue/resource conservation and conflict constraints should remain explicit rather than be learned as soft behavior.
- Observations need protocol/service, radio, topology, and context groups, each with timestamp, sampling rate, missingness mask, confidence, and configuration metadata. A recommended hybrid state keeps queues/deadlines/AoI/HARQ/graph explicit, while CSI/traffic/mobility are latent beliefs and uncertainty is explicit.
- Action encoding should preserve combinatorial structure: assignments as bipartite/matching graphs, conflicts as graph masks, and continuous power separated from discrete schedules. The model should predict more than the next observation: service-rate distributions, P95/P99 delay, packet loss, AoI, energy, fairness, constraint violations, and OOD score.
- The offline/online loop is: combine logs, simulation, ray tracing, protocol rules, and multi-policy data; train encoders, belief updates, action-conditioned dynamics, outcome heads, and uncertainty; online update asynchronous belief, generate candidates, solve/filter, act, then recalibrate from realized KPIs.
- The review's nine-step route prioritizes objective/constraint specification, a data contract, structured state, 1/5/20-step dynamics and outcome prediction, offline counterfactual validation, a planning interface, independent safety shell, progressive deployment, and continuous drift/fallback governance.
- Evaluation has five layers: representation (task sufficiency/topology preservation/compression), dynamics (1/5/20-step error, NLL, calibration/coverage), planning (return, violation, regret, CVaR), system (throughput, P95/P99 delay, AoI, energy, fairness), and deployment (drift, fallback, online calibration, auditability). It requires matched data, interaction, inference-latency, and feasible-action budgets.
- The review explicitly warns that low one-step error can reverse action ranking, and average KPI accuracy can hide P99 delay or violation risk. Action ranking, top-k regret, constraint classification, risk calibration, rare-event recall, cross-topology/load/band/site transfer, and closed-loop tests are therefore necessary.
- Recommended architecture is composable: specialist radio/CSI, traffic, mobility, fault, queue, and protocol models at the bottom; graph-aligned belief composition, MPC/solvers/policies in the middle; intent/governance at the top. Every module interface should declare state/action schema, time scale, conditioning variables, predictive distribution, calibration, validity domain, and version.
- The near-term acceptance criterion is a rigorous single-task closed loop with open data/code, strong heuristic and analytic baselines, matched-budget evaluation across load/topology classes, calibrated uncertainty, and latency/shadow-mode evidence; a monolithic "6G world model" is explicitly not the default goal.

### Auxiliary PPT and speaker notes

- The 10-slide PPT and Markdown speaker notes are a presentation of the same review, not an independent survey. They add no contradictory definition.
- Their project-facing synthesis is: channel map/radio field provides CSI, geometry, blockage, and propagation; explicit MAC/task state provides queues, deadlines, AoI, HARQ, and resource occupancy; a dynamic AP/UE or network topology graph captures adjacency, interference, and conflicts; the world model rolls out candidate joint actions; a solver and independent safety shell enforce feasibility and fallback.
- The notes explicitly warn that the channel map is an environmental representation module, not a complete MAC world model. They identify the three open interfaces as alignment of channel maps with entities/timestamps, compositional encoding of association/resource actions, and action-ranking validation on unseen topology/load/blockage.

### Current evaluation implementation versus review

- Existing metric aggregation includes link-activity precision/recall/F1/AP/AUC, all-link rate RMSE, active-link rate RMSE, task-state RMSE, and probability calibration Brier/ECE for link activity. It is more than a single RMSE, but it remains prediction-centric.
- Missing mainline metrics are service-rate distributions, P95/P99 delay, packet/task loss, completion/deadline violation, AoI, energy, fairness, action sensitivity/ranking regret, constraint classification, uncertainty coverage, OOD transfer, online latency, fallback rate, and closed-loop utility.
- The review's queue law and outcome-head design should not be copied into the project until raw AirFogSim logs can support the fields. The immediate methodological decision is which subset is observable and which quantities must be added by instrumenting the simulator.
- The review references both peer-reviewed foundations (World Models/Dreamer/MuZero/TD-MPC2, CSI-JEPA and selected wireless work) and many 2025-2026 preprints/frameworks. Quantitative claims from these categories must remain evidence-labeled and cannot be ranked directly.

### Current PI-JWM versus the review: initial gap map

- The current project document defines the direct research question correctly as action-conditioned joint state transition, but the implemented v6/v8 data contract is narrower than the review's MAC contract.
- Current implementation facts: history length 8, future horizon 3, 58 nodes, 314 directed candidate edges; node features are position/velocity/acceleration plus CPU/storage; physical edge features are derived from endpoint states; information edge features include distance, rate sum, mean CSI, active-task count, and allocated-RB count.
- The current task state is a 9-dimensional global aggregate (counts, total data/CPU, mean deadline/priority, stage counts), not per-task queues/tokens with arrival/completion/deadline identities. It cannot strictly yield per-task mean/P95 delay, deadline violation, or task-level energy.
- The current action is a 6-dimensional edge aggregate (offload count, RB-task count/total, CPU-task count/total, return count), not a combinatorial user/RB/slot matching action. This limits genuine counterfactual action coverage and makes it unlike the review's explicit MAC action schema.
- v8 predicts node state, link activity, link rate, and aggregate task state with supervised MSE/BCE-style multitask losses. It does not yet have calibrated probabilistic dynamics, explicit queue/resource conservation, P95/P99 outcome heads, energy head, fairness head, or OOD-risk head as a unified main objective.
- Current v8 graph modeling is a useful structured dual branch but should not be described as the review's full hybrid belief: explicit protocol state, latent radio/traffic state, topology/conflict graph, and uncertainty are not all represented with the review's declared interfaces.
- The redesign therefore should not start by replacing AirFogSim or assuming cell-free. It should first decide the control task and time scale, then expand the data contract only where raw simulator/log fields support it, and separate explicit legal dynamics from learned uncertain components.

### Dataset schema cross-check

- The archived baseline dataset `world_model_dataset_v0` has 950 samples, 8 history steps, 3 future steps, 37 nodes, and 188 directed candidate edges.
- The active-heavy 60-seed dataset used by the latest evidence has 23,400 samples, 8 history steps, 3 future steps, 58 nodes, and 314 directed candidate edges. It still exposes the same main arrays: `x_node[...,7]`, `x_link[...,5]`, `x_task[...,9]`, `edge_a_hist[...,6]`, `edge_a_future[...,6]`, `y_link_rate`, `y_link_active`, and `y_task[...,9]`.
- The active-heavy summary reports active-link item ratio about 0.00147, so rate/activity prediction is severely imbalanced. This supports the review's warning that activity, service, tail risk, and calibration need separate treatment rather than one unqualified link-rate RMSE.
- The current data contract has no visible per-task queue/DAG tensor, no direct P95/P99 delay target, no actual energy target, and no explicit uncertainty target in the main world-model arrays. Any redesign claiming those quantities must first extend the data contract or label them as downstream-only unavailable quantities.

### Immediate redesign implication

- The review supports retaining a joint action-conditioned PI-JWM, but the project should be reframed around a declared control slice rather than a vague all-network objective. The first candidate slice is task/service-aware edge scheduling over the existing V2I/V2U graph, with explicit feasibility and a measurable service outcome.
- A full MAC-style contract would require adding or recovering per-task queue/arrival/deadline/service fields, action semantics at the task/RB/CPU level, resource capacities/occupancy, exogenous-versus-controllable decomposition, and uncertainty/risk labels. These are decisions to confirm after the Chinese review is unlocked, not yet implemented changes.

### Chinese review extraction status

- The Chinese PDF became readable after it was copied to an ASCII temporary path and processed with `pdftotext -enc UTF-8 -layout`; direct `pypdf` extraction had a font mapping issue and was discarded for semantic reading.
- It has 36 pages and the same English/Chinese bilingual review structure. The first page confirms the same conclusion: a world model should learn a compact internal world that predicts task consequences under actions, with explicit protocol state, latent wireless/traffic beliefs, graph topology, and uncertainty.
- The Chinese version has now been read through the communication-network, MAC, evaluation, safety, architecture, roadmap, conclusion, and reference sections. It adds no substantive contradiction to the English version.
- The two PDFs are language versions of one structured scoping review, not two independent reviews or two independent sources of evidence. The PPT and speaker notes are also derivatives of the same synthesis.
- Neither PDF contains a `cell-free`/`无蜂窝` scenario requirement. It discusses cells/base stations only as examples and leaves topology choice task-dependent.
- Neither PDF uses `channel map`/`信道图谱` or path-loss prediction as the governing objective. It treats geometry, propagation, CSI-like measurements, and service outcomes as distinct layers; radio-field/channel representations are optional specialist modules and intermediate targets.
- The review itself has no visible author list or peer-reviewed venue and cites a mixture of peer-reviewed papers, preprints, frameworks, and 2026 material. It is useful as a design synthesis, but individual claims must be grounded in the cited primary literature before becoming project definitions or proofs.

### Review-to-PI-JWM decision boundary

- Directly applicable: retain action-conditioned joint dynamics; explicitly separate exogenous and controllable state; preserve physical/task topology; keep resource/queue legality explicit; add outcome and uncertainty interfaces; evaluate rollout, action ranking, constraints, tails, transfer, and latency in addition to one-step error.
- Applicable only after data-schema work: per-task queue/HOL delay/deadline/AoI/HARQ state, task/RB/CPU matching actions, service-rate distributions, P95/P99 delay, packet/task loss, actual energy, and constraint-risk labels.
- Not justified by the review: declaring PI-JWM cell-free, replacing the project with CSI or path-loss prediction, treating a channel map as the whole world model, using every possible MAC action at once, or claiming real-network control from AirFogSim-only evidence.
- The first redesign decision must therefore be a narrow control slice and time scale. The current AirFogSim topology supports a candidate V2I/V2U task-offloading/resource-allocation slice, but this remains a project choice rather than a conclusion imposed by the review.

## 2026-07-31 Multiseed Data Findings

- AirFogSim's `TaskManager.getOffloadingTasksWithNumber()` aliases `_offloading_tasks` lists and extends them with `_returning_tasks`. When both maps share a node key, a read mutates runtime state and duplicates the same task object on every call. PI-JWM must use a copied transmission view for reproducible data generation.
- Seeds 0/1/2 produce different graph and task sizes while each covers all nine V/U/I directed channel families. Fixed-size training tensors therefore require explicit padding, masks, identity ordering, and flow matching; the old fixed `x_link` assumption cannot be reused silently.
- Across the three development seeds, task completion is 0.690867 +/- 0.133122, P95 successful delay is 0.990000 +/- 0.190000 seconds, and task/CPU/energy conservation violations are zero.
- Native AirFogSim DAG edges still have zero explicit payload coverage. They are valid precedence supervision but cannot be relabeled as `dependency_data` communication.
- Uncertainty coverage, action regret, and OOD transfer remain not computable from factual simulator trajectories alone.

### Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| PowerShell inline `(if(...))` expression failed while reporting extraction metadata | 1 | Remove the expression from the reporting command; it did not affect the first PDF extraction |
| Direct `pypdf` read of the Chinese PDF returned `PermissionError` | 1 | Copy the file with PowerShell to an ASCII temporary path, then extract the copy |
| PowerShell piped a `foreach` expression directly after its closing brace and reported an empty pipe element | 1 | Materialize the term-count rows first, then format the array; no document content was affected |

## 2026-08-01 Sparse-Event Diagnostic Findings

- The development training labels are strongly imbalanced: link activity positive rate is 0.007538 and flow presence positive rate is 0.008899; both positive weights reach the configured cap of 50. Task presence is less sparse at 0.391950 with weight 1.551345.
- Positive weighting materially improves sparse link ranking: balanced AUPRC is 0.044855 on seed 1 and 0.045419 on seed 2, compared with 0.018502/0.020042 without weights and 0.031699/0.022387 for last persistence.
- The improvement is not a general rollout win. Last persistence remains better on link F1, active-only rate MAE, task-presence F1, and lifecycle macro-F1 on both development evaluation seeds.
- A five-epoch learned model can beat the zero-activity baseline while still underperforming a simple temporal rule. Future method claims must therefore retain both simple baselines, not compare only against all-zero prediction.
- The next bottleneck is explicit temporal continuity and task-state evolution, not immediate JEPA coupling. JEPA comparison remains a later controlled ablation after the base model clears the persistence gate.
# 2026-08-02 Formal dual-graph CPU findings

- The formal 54-trajectory tensor is sufficient to exercise strict physical-edge, information-flow, CIP, CFE, task/DAG, and 8-dimensional logged-action interfaces without touching locked-test.
- A 64/32/32 CPU comparison is enough to verify fair method plumbing, but not enough to establish converged prediction quality.
- Coupled dual-graph rollout improved short-run link F1 over persistence (validation 0.187500 vs 0.100840; calibration 0.250000 vs 0.132841) while remaining much worse on node x MAE (34.078557/32.334389 m vs 1.418819/2.426800 m).
- Therefore the defensible state is `cpu_smoke_ready=true`, not `formal_training_ready=true`; long GPU training, multiple seeds, ablations, and complete paper-method baselines remain necessary.

## 2026-08-03 Unified R0-R9 Roadmap Audit

- Reusable without rerunning: 60 raw AirFogSim trajectories, the 54 non-locked trajectory split, logged unload/RB/CPU actions, task/resource/system-result sidecars, metric implementations, CPU/GPU runner patterns, checkpoint reload checks, and implementation-level graph tests.
- Reusable only after semantic adaptation: directed aggregation, concat/gated/cross-attention fusion code, action-conditioned recurrent rollout, masked multitask loss, uncertainty/metric interfaces, and simple prediction baselines.
- Must be regenerated under the advisor-aligned ontology: physical/information graph mapping, `CIP/CEP/CFL` tensors, explicit-plus-latent dual-state inputs/targets, all formal checkpoints, all module-ablation numbers, and all final baseline/closed-loop comparisons.
- Existing v6/v7/v8 ablations and 2026-08-02 CPU/GPU experiments narrow implementation choices but use the transition ontology; they cannot prove the new method.
- GPU is unnecessary for ontology, contract conversion, integrity checks, metric freezing, simple rule/persistence baselines, feasibility masks, statistics, and report generation. GPU is required for converged multi-seed module ablations, finalist world-model training, learned-policy training, and complete learned-paper baselines.
- Pretraining and cross-scenario transfer are explicitly outside R0-R9; all current candidates start from random initialization on the frozen AirFogSim train split.

## 2026-08-03 Strict Dual-Graph Literature Findings

- Yağan et al. (IEEE TPDS 2012, DOI `10.1109/TPDS.2012.62`) directly support the structural principle of two interacting networks with their own intra-layer edges and explicit inter-network dependency links/matrix. They do not define PI-JWM's mobile spatial-neighborhood rule or task semantics.
- Shen et al. (IEEE JSAC 2021, DOI `10.1109/JSAC.2020.3036965`) explicitly model channel states as wireless communication-graph edge features. This directly supports moving CSI/channel/interference semantics to PI-JWM information edges, but the paper is a single wireless RRM graph rather than a full physical-information world model.
- No single verified top-journal paper defines PI-JWM's complete combination. The defensible statement is a transparent synthesis: TPDS anchors the dual-network structure, JSAC anchors wireless edge semantics, and PI-JWM-specific choices remain subject to controlled ablation.

## 2026-08-03 R1 Teacher-Aligned Contract Findings

- The 54 non-locked saved AirFogSim trajectories contain every field required to instantiate the advisor-aligned v3 graph/tensor contract, so no simulator rerun is required for R1.
- Physical edges are now directed same-slot spatial relations with geometry-only features; information edges are directed communication-interface links with pre-action channel state, action allocation, and post-action outcome groups. `CIP/CEP` are cross-graph relations, while `CFL` is a separate flow-to-link business relation; `CFE` is deprecated.
- Optional radio-detail fields absent from current AirFogSim records remain represented by false feature masks. Their absence limits later high-fidelity channel claims but does not invalidate R1's required topology, action, task, or observed-rate supervision.
- Waiting business flows must be materialized from their first recorded time, before RB service begins; otherwise the graph would incorrectly erase queued information. The v3 tensorizer now preserves this waiting phase and tests it explicitly.
- A naive normalization pass that concatenates all 36 training trajectories exceeds available memory. Streaming per-feature count/sum/squared-sum produces the same statistics with bounded memory and is now protected by a regression test.
- The locked-test directory layout differs from the unlocked layout. R1 reads only each locked trajectory's manifest from `locked_test/trajectories/`, emits no locked tensors, and records `label_content_read=false` and `tensorized=false`.

## 2026-08-03 R2 Evaluation Protocol Findings

- The v3 trajectory evaluator reports 16 directly computable teacher-state/event metrics and retains three uncertainty rows as explicit N/A; the existing factual sidecar can report 22 system/resource/safety metric names. Action regret, OOD fallback, and deployment latency also remain unavailable until the corresponding counterfactual, transfer, or timed-inference evidence exists.
- Physical-edge registry sources are limited to distance and relative speed; information-link activity and rate are sourced from `information_edge_state.outcome.active_task_count` and `information_edge_state.outcome.rate_sum`. The evaluation layer therefore does not reintroduce the rejected channel-as-physical-edge semantics.
- Last persistence is a necessary minimum gate. Across 54 non-locked trajectories it reduces position RMSE from 675.81 m (zero state) to 43.56 m and raises task-lifecycle macro-F1 from 0.0154 to 0.7805, while information-link activity remains difficult (macro F1 0.0702, AUPRC 0.00961) and active-only rate MAE remains high (473.42 Mbps). These are sanity baselines, not trained-model or formal method claims.
- A metric row is never silently dropped or replaced by zero: every result is either finite with its audit quantities or `not_computable` with a reason. This is especially important for deterministic baselines and factual-only AirFogSim trajectories.
- R2 used CPU only and accessed no locked labels. The frozen protocol is now an input constraint for R3-R9 rather than a tuning surface.

## 2026-08-03 R2 Pre-Revision Correctness Audit — Preliminary Findings

- The 36-row registry and the 22-row factual sidecar are currently stored in separate namespaces without a machine-readable canonical mapping. Several direct pairs are obvious (`task_completion_rate`→`system.task_completion_rate`, `information_throughput`→`system.application_throughput`), but task failure, priority-weighted completion, dependency coverage/delivery, and the legacy `physical_link_active_ratio` have no frozen registry destination. This would create reporting ambiguity in R8/R9 and must be resolved inside R2.
- `physical_link_active_ratio` is a legacy AirFogSim/source name. Under the teacher-aligned ontology its wireless activity meaning belongs to an information-link KPI; retaining the raw source name is acceptable only if the protocol maps it to a canonical information-link metric and marks the old name as a source alias, not as the PI-JWM graph definition.
- The checkpoint rule references virtual metric `state.required_continuous.normalized_rmse`, but R2 does not yet define its components, train-only scales, missing-component rule, or exact aggregation. Allowing an unregistered virtual term makes checkpoint selection non-reproducible and is a confirmed R2 protocol gap.
- Statistical language currently says “macro over seeds” while the baseline `seed` field is an AirFogSim environment trajectory seed; future learned comparisons add an independent training-initialization seed. These two levels must be named and aggregated separately to avoid pseudo-replication and ambiguous confidence intervals.
- The zero/last evaluator correctly reads physical geometry from physical edges and wireless activity/rate from information edges. It also keeps deterministic uncertainty outputs as explicit N/A. No local evidence has yet shown a channel-as-physical-edge regression in R2.
- The prior `main_experiment_contract.py` and `formal_world_model_metrics_v1.py` retain old physical-channel terminology. They are historical interfaces and must not be imported as the teacher-aligned R3 source of truth without semantic adaptation.

### Primary Literature Evidence Collected (round 1)

- **Sparse-event evaluator:** Saito and Rehmsmeier, PLOS ONE 2015, DOI `10.1371/journal.pone.0118432`, directly support using precision–recall analysis for severely imbalanced binary events; this supports retaining AUPRC beside thresholded F1 for information-link activity. It does not choose PI-JWM's activity threshold or averaging hierarchy.
- **Probabilistic output:** Gneiting and Raftery, JASA 2007, DOI `10.1198/016214506000001437`, establish proper scoring rules and the logarithmic score; Gneiting, Balabdaoui, and Raftery, JRSS-B 2007, DOI `10.1111/j.1467-9868.2007.00587.x`, frame forecast quality as sharpness subject to calibration. Together they support NLL plus coverage/width rather than coverage alone.
- **Prediction intervals:** Romano, Patterson, and Candès, NeurIPS 2019, `Conformalized Quantile Regression`, support a separate calibration set, finite-sample marginal coverage, and reporting interval efficiency/width. The guarantee requires the method's assumptions (notably exchangeability); PI-JWM must not claim unconditional time-series or cross-scenario validity.
- **Resource fairness:** Jain, Chiu, and Hawe, DEC-TR-301 (1984), define the bounded fairness index for nonnegative resource allocations. This supports Jain fairness over a fixed, explicitly defined source-service population; it does not justify changing the population per method or using an undefined zero denominator.
- **Few-run learned-method statistics:** Agarwal et al., NeurIPS 2021, `Deep Reinforcement Learning at the Edge of the Statistical Precipice`, support reporting uncertainty rather than only point estimates in few-run comparisons. This strengthens the need to distinguish training replicates from environment trajectories and to report paired interval estimates.
- **Communication-system KPI scope:** ITU-T Y.1540 (12/2019) defines IP transfer performance parameters for speed, accuracy, dependability, and availability. It supports keeping delay/throughput/reliability as separate observable system KPIs; it does not determine AirFogSim task-level formulas or units.

### Primary Literature Evidence Collected (round 2)

- **Model selection leakage:** Cawley and Talbot, JMLR 2010, show that optimizing a noisy model-selection criterion can overfit the selection procedure and bias later performance estimates. This directly supports keeping validation-based selection separate from calibration and one-shot locked testing; it also argues against tuning the checkpoint composite after seeing evaluation results.
- **Repeated learned-agent evidence:** Patterson et al., JMLR 2024, survey empirical design in RL, including variation, stability, hypothesis testing, baselines, hyperparameters, and experimenter bias. This supports explicit training-replicate IDs, failed-run retention, shared budgets, and transparent hyperparameter search rather than treating one environment seed as a training replicate.
- **Multiple environments/methods:** Demšar, JMLR 2006, supports paired non-parametric comparisons across independent data sets for multiple classifiers. Its assumptions do not directly turn correlated AirFogSim trajectories or training seeds into independent data sets; PI-JWM therefore needs an explicit hierarchy rather than blindly applying a Wilcoxon/Friedman test.
- **Continuous forecast errors:** Hyndman and Koehler, International Journal of Forecasting 2006, document scale and comparability problems in forecast accuracy measures. This supports reporting physical-unit MAE/RMSE per field and using an explicitly train-scaled composite only for checkpoint selection, not as the sole scientific result.
- **JEPA reference status:** Bou Chaaya, Girgis, and Bennis, IEEE TWC vol. 25 (2026), DOI `10.1109/TWC.2025.3644600`, is verified as the cited joint-embedding/multimodal latent-dynamics resource-planning paper. It supports JEPA-style coupling as an R4 candidate, not the correctness of R2's metric mapping or a decision to freeze JEPA now.
- **Wireless system objectives:** recent IEEE resource-allocation papers continue to separate latency, energy, queue stability/reliability, throughput, fairness, constraints, and runtime rather than collapse all quality into state RMSE. This supports the layered R2 registry, while AirFogSim-specific numerator/denominator definitions still require local source-code proof.

### Independent Code Review Findings

- Verdict was **not ready without fixes**. The reviewer independently confirmed the three critical gaps: no unified factual/baseline row schema, legacy physical-link wireless activity still exposed without canonical information-link remapping, and a non-executable checkpoint composite.
- Important confirmed gaps: persistence ignored history masks; exact dataset/integrity set equality was not enforced; the factual CSV was filtered only after reading; pooled micro/CI/effect-size rules were not implemented; protocol validation was easy to bypass; AP/F1/lifecycle conventions were incomplete; output manifest lacked input/code/environment provenance; tests missed these gates.
- Minor but real gaps: environment seed was named generically as `seed`, and the CLI module docstring followed the future import so `__doc__` was empty.
- The review reproduced the current historical-mask bug and recomputed the current 54/6/108 counts and output hashes. It did not modify project files.

### Required R2 Corrections

1. Bind all 22 factual source metrics to canonical registry IDs and emit 1,188 factual rows in the same long-form result schema as prediction baselines.
2. Recompute canonical information-link active ratio from v3 information-edge tensors; retain `physical_link_active_ratio` only as a documented source alias, never as current graph semantics.
3. Register and fully define the ten-component train-scaled checkpoint metric; bind train-stat hash, environment split IDs, weights and transforms; validate them structurally.
4. Make last-persistence require observed previous state, and freeze AP ties, event threshold calibration, and fixed five-class lifecycle macro-F1.
5. Separate `environment_seed` and `training_seed`, implement macro plus additive pooled micro, mark `n<2` sample std/CI as N/A, and freeze paired complete-case comparison rules.
6. Validate exact dataset-summary/index/integrity set equality before tensor reads; verify factual input provenance before parsing; record input/code/runtime hashes in the output.

### R2 Correctness Audit Resolution

- All six required corrections are now implemented. The registry has 43 canonical entries; all 22 factual source metrics map one-to-one into it, and the unified long-form artifact contains 1,188 factual rows plus 2,160 prediction-baseline rows.

### R2 Final Semantic-Binding Resolution

- The last reproducibility gap was not metric arithmetic but tensor meaning: a valid NPZ hash cannot prove that column order still matches the code importing those columns. R2 now verifies the R1 tensor contract and graph protocol through the upstream manifest and then compares all four graph feature lists against the imported constants in exact order.
- A negative test changes feature order and updates the manifest hash; the build still fails on semantic mismatch. This separates byte integrity from schema integrity.
- The final canonical bundle verifies 62 inputs (54 trajectory NPZ files plus eight source metadata/metric files), six actual code dependencies, and 13 generated outputs. Independent review returned `Ready`; no known R2 semantic or protocol blocker remains.
- `system.information_link_active_ratio` is recomputed from v3 `information_edge_state.outcome.active_task_count` with information-edge presence/feature masks. Across all 54 non-locked trajectories it exactly matches the legacy source value (maximum absolute difference 0), while its graph semantics and provenance are now correct.
- Continuous zero baselines use train-only feature means. Persistence requires a valid previous observation; lifecycle macro-F1 uses the frozen five-class label space; `n=1` sample standard deviation and interval are N/A rather than zero; additive metrics retain pooled numerators/denominators.
- The checkpoint term is now the registered `selection.required_continuous.normalized_error`: ten frozen continuous groups, explicit train-only scales, aggregate-before-combine semantics, four exact 0.25 checkpoint weights, and a calibration-only event-threshold policy.
- The bundle verifies upstream manifests before parsing inputs, checks exact dataset/index/locked identity sets, records code/runtime/git/input provenance, and emits 13 manifest-managed files. The audited candidate validates 54 non-locked trajectories, 6 sealed locked trajectories, 108 baseline reports, and 3,348 unique canonical result rows with no failed gate.
- Primary-source support is now explicit: Saito–Rehmsmeier for imbalanced-event PR evaluation; Gneiting–Raftery and Gneiting–Balabdaoui–Raftery for proper scores/calibration/sharpness; Romano–Patterson–Candès for separate conformal calibration and coverage/width; Cawley–Talbot for model-selection leakage; Jain–Chiu–Hawe for fairness; ITU-T Y.1540 for communication performance quantities; Patterson et al. for empirical-design discipline.

### R2 DAG Column-Semantics Finding and Resolution

- A three-value statistic is insufficient if the consumer reads the wrong column. R1 freezes `task_dag_state` as `parent_count`, `unfinished_parent_count`, `release_ready`; all consumers must resolve the desired name from `FORMAL_DAG_STATE_FEATURES`, not hard-code index 0.
- The prior canonical R2 artifact mislabeled parent-count baseline values and scale as unfinished-parent-count. The corrected evaluator reads index 1 and the corrected checkpoint scale is `1.4639630895` rather than `1.9266825090`.
- Regression fixtures must assign visibly different values to semantic columns. All-zero fixtures and dimension-only assertions cannot detect this class of error.

### R3 CPU Preflight Findings

- The old `directed_dynamic_v2_1` ontology cannot be relabeled as the advisor-aligned v3 method. R3 therefore reuses only generic directed relation operators and implements new v3-native physical/information branches and `CIP/CEP/CFL` coupling.
- Explicit state and latent belief are both executable: the former supports auditable physical-unit targets, while the latter carries action-conditioned rollout state. The CPU reference uses deterministic GRU dynamics solely as a contract probe; it does not select the final R4 dynamics.
- Activity targets must be derived from raw `information_edge_state.outcome.active_task_count` before continuous normalization. Thresholding a normalized value would change the event semantics.
- The same-parameter coupling ablation is essential. Comparing two independently updated models would confound parameter drift with the presence of `CIP/CEP/CFL`.
- R3 establishes `r3_cpu_preflight_ready=true`, not superiority or convergence. Graph-RSSM, Transformer, JEPA-style coupling, probabilistic heads, and final policy architecture remain R4-R6 experimental candidates.

## 2026-08-04 R3 Literature Re-Audit — Initial Claim Inventory

- The R3 design correctly separates three evidence levels: executable contract, literature-supported general principle, and PI-JWM-specific hypothesis. It does not currently claim that deterministic Graph-GRU, gated coupling, JEPA, RSSM, or Transformer is the final method.
- Claims requiring primary-source verification are: separate relation-aware graph encoders; explicit inter-network relations; latent belief for partial observability; action-conditioned open-loop latent rollout; separate explicit reconstruction/prediction heads; mask-safe learning with missing/padded entities; and strict train/validation/calibration/test boundaries.
- `CIP/CEP/CFL`, four named latent groups, the exact seven continuous targets, and the choice to combine explicit and implicit state are PI-JWM contract/design choices. Literature can support their plausibility and evaluation discipline, but cannot prove these exact choices correct without R4-R8 experiments.
- Current R3 source uses deterministic GRU dynamics only as a reference execution probe. The main document already labels RSSM, Transformer, JEPA coupling, probabilistic heads, and policy architecture as candidates rather than completed methods.
- The design document's promised artifact filenames (`config.json`, `window_selection.json`, `smoke_metrics.json`, `checkpoint_reload_report.json`, `validation_report.json`) do not match the implemented canonical filenames (`preflight_summary.json`, `selected_windows.json`, `objective_reports.json`, `rollout_checks.json`, `metric_interface_report.json`). This is a documentation-contract mismatch that must be corrected or explicitly versioned during the audit.
- The current loader keeps historical states but discards historical `task_action`; only rollout-horizon actions are placed in `future_action`. Therefore the design phrase “observation–action history generates the current belief” is not implemented literally. Either R3 must encode historical actions or the claim must be narrowed to “historical states, which already contain executed-action consequences.” The former is more consistent with standard action-conditioned state-space models and should be evaluated as a corrective candidate.
- The reference model encodes the three-dimensional per-task DAG summary but does not run message passing over `dag_edge_index/dag_edge_present`; it also does not use `task_information_node_index` during belief inference. R3 may truthfully claim “DAG summary input,” but not “DAG graph propagation” or full task-agent relational encoding.
- The recurrent rollout carries only latent state. Explicit predictions are audit/learning heads and are not fed into the next step. The roadmap sentence saying both explicit and implicit states are passed to the next step conflicts with the implemented and standard latent-state rollout; the theory text must choose one interpretation explicitly.
- The graph-update masks remain fixed at the last observed presence throughout rollout; predicted presence logits do not gate later messages. This is acceptable only as a minimal R3 execution reference, not as a faithful dynamic-topology model. R4 must compare or implement predicted-presence-conditioned topology before final-method claims.

### Primary-source evidence — latent dynamics and graph simulation

- PlaNet (Hafner et al., ICML 2019, PMLR 97) defines an action-conditioned transition `p(s_t|s_{t-1},a_{t-1})`, observation model `p(o_t|s_t)`, and inference state `q(s_t|o_{<=t},a_{<t})`; it also evaluates multi-step latent predictions. This directly supports latent open-loop action-conditioned rollout and an explicit observation/state head, and it exposes the current R3 omission of historical actions from belief inference.
- Dreamer (Hafner et al., ICLR 2020) rolls the RSSM forward for many steps using only actions after a short observed context. It supports R3's rule that future labels must not enter rollout. It does not prove deterministic GRU is the right PI-JWM dynamics or that explicit decoded states must be fed back.
- Graph Network-based Simulators (Sanchez-Gonzalez et al., ICML 2020, PMLR 119) represent entities as graph nodes and compute learned dynamics by message passing, demonstrating long-rollout graph simulation and warning that accumulated rollout error matters. Graph Networks as Learnable Physics Engines (ICML 2018) similarly supports object/relation-centric transition models. These works support separate relation-aware graph encoders in principle, but not the exact PI-JWM two-layer ontology or `CIP/CEP/CFL` definitions.
- Bou Chaaya, Girgis, and Bennis (IEEE TWC 2026, DOI `10.1109/TWC.2025.3644600`) is verified through the Oulu institutional repository and DOI metadata. It learns joint control and wireless dynamics in latent space for resource planning, supporting modality-specific encoders and causal latent conditioning as an R4 candidate. Its setup does not directly validate bidirectional PI-JWM graph coupling, task/DAG propagation, or the current deterministic GRU reference.

### Primary-source evidence — wireless graph semantics and inter-network coupling

- Shen et al. (IEEE JSAC 2021, DOI `10.1109/JSAC.2020.3036965`) explicitly define a directed wireless graph whose nodes are agents, whose directed edges are direct communication or interference links, and whose edge features carry the corresponding channel state. Their message-passing update consumes neighboring node state together with the directed edge feature and is permutation equivariant. This is direct support for PI-JWM placing wireless channel/link state on **information edges** and using relation-aware message passing; it is not support for putting CSI on physical spatial edges.
- Yağan et al. (IEEE TPDS 2012, DOI `10.1109/TPDS.2012.62`) study two interacting networks and explicit interconnecting links between their nodes. This supports representing physical and information networks as separate intralayer graphs joined by explicit cross-layer relations. It does not establish PI-JWM's exact `CIP/CEP/CFL` meanings or the direction/parameterization of learned coupling.
- Bou Chaaya et al. use two coupled JEPAs: a control JEPA for control-transition dynamics and a wireless JEPA for CSI dynamics, with cross-modal conditioning from control to wireless. This is strong evidence that modality-specific latent dynamics plus explicit cross-modal conditioning is a technically legitimate R4 candidate. Because their coupling is problem-specific and directional, it cannot be cited as proof that PI-JWM's three cross-relations, bidirectionality, or gated fusion is optimal.
- The evidence boundary is now explicit: JSAC supports information-edge wireless semantics; TPDS supports separate interacting networks plus interlinks; TWC supports coupled modality-specific latent dynamics. No verified paper defines PI-JWM's complete ontology or proves its exact coupling implementation. Those exact choices remain hypotheses to be selected by R4 controlled ablations.

### R3 primary-source audit — exact claim boundaries

| R3 statement | Primary-source support | What the source does not prove | Audit disposition |
| --- | --- | --- | --- |
| Current belief must depend on observation and past-action history | PlaNet, ICML 2019, defines `q(s_t | o_{<=t}, a_{<t})` and an action-conditioned transition | It does not prescribe PI-JWM's graph fields or deterministic reference GRU | **Required interface; implementation corrected to retain and encode historical actions** |
| Candidate actions drive open-loop latent rollout; future labels do not enter recurrence | PlaNet's `p(s_t | s_{t-1},a_{t-1})` and Dreamer's latent imagination support this pattern | They do not prove a 20-step horizon is accurate for AirFogSim | **Required execution property; existing no-future-target test retained** |
| Graph entities and relations should be encoded by message passing | Graph Network-based Simulators, ICML 2020, represent entities as nodes and compute dynamics by learned message passing | It does not define a physical-information dual graph, wireless edges, or `CIP/CEP/CFL` | **Supported general mechanism; exact PI relations remain experimental** |
| Wireless channel/link state belongs on directed information edges | Shen et al., IEEE JSAC 2021, define wireless agents as nodes and direct communication/interference links as directed edges carrying channel state | It is a single wireless RRM graph, not a PI world model | **Frozen semantic rule** |
| Two intralayer networks may be joined by explicit cross-network links | Yağan et al., IEEE TPDS 2012, model two interacting networks with interconnecting links | It does not specify PI-JWM relation names, directionality, or neural fusion | **Frozen structural principle; coupling parameterization remains experimental** |
| Separate modality encoders and latent cross-modal conditioning are credible candidates | Bou Chaaya et al., IEEE TWC 2026, couple control and wireless JEPAs; control latent conditions wireless latent prediction | Their coupling is directional, assumes weak inter-device CSI interaction, and does not cover PI-JWM queues, DAGs, CPU or bidirectional graph relations | **R4 candidate only; not frozen as the method** |

### R3 implementation re-audit — corrections and preserved limits

- Historical `task_action`, action presence, and task-to-information-role indices are now retained in a separate `history_action` namespace. A masked temporal action encoder contributes to the per-task current belief, matching the observation-action history claim without mixing past actions with candidate future actions.
- R3 batch construction now rejects malformed physical/information endpoint arrays, partially padded endpoint pairs, out-of-range `CIP/CEP/CFL`, missing mappings for active entities, and malformed or out-of-range historical/future action-to-information indices. Bad relations are no longer silently converted to zero messages.
- The corrected formal CPU candidate again passes train/validation/calibration windows at horizons 1/5/20 for coupled and same-parameter no-coupling controls. The new execution deltas are `action_delta=0.0469472557` and `coupling_delta=0.0330387801`; they remain path-execution diagnostics, not accuracy or superiority evidence.
- The model still encodes the three-value DAG summary per task but does not message-pass over `dag_edge_index`. It may be described only as **DAG-summary-aware** in R3.
- The reference rollout still propagates latent state while saving explicit predictions for loss, metrics, constraints and audit. Decoded explicit predictions are not recursively re-encoded, consistent with PlaNet-style latent rollout. Documentation saying both tensors enter the neural next-step transition must be corrected.
- Presence heads predict future entity/link presence, but the R3 reference graph operator keeps the last observed presence mask during rollout. Thus R3 proves typed output and latent recursion, not fully learned topology recursion. Predicted-presence-conditioned messages must be implemented and ablated in R4 before a dynamic-topology final-method claim.
- Zotero evidence records are verified locally and in the cloud: PlaNet `WRX5XKKD`, Graph Network-based Simulators `W79FZZKX`, wireless GNN `65HG2GWI`, dual-network structure `C4M4482F`, and coupled multimodal JEPA `VSAI6T23`.

### R3 re-audit final verdict

- **Verdict: Ready within the R3 execution boundary.** The claim is limited to contract-correct CPU execution, not method accuracy, convergence, superiority, learned topology recursion, DAG-edge propagation, calibration or closed-loop benefit.

### R4 CPU Interface Findings

- The existing R3 API already supplies the required common contract: `ExplicitStateBatch` input and `predicted_explicit`/`predicted_logits`/`predicted_belief` output. R4 should wrap and extend this contract rather than alter R3.
- The current checkout is a `codex/` feature branch, but the R3 implementation is an uncommitted workspace asset. A new worktree from the current commit would omit the exact R3 files that R4 must reuse, so R4 is being added conservatively in place without moving or cleaning unrelated changes.
- A machine-readable registry needs separate `reference`, `planned`, `executable`, `reserve`, and `deferred` states. Only `reference` and actually implemented `executable` components may enter the CPU model factory; literature candidates are not executable merely because they are listed.
- Controlled R4 screening must reject configurations that change more than one module family. This is enforced before construction rather than inferred later from experiment names.
- Fresh R3 verification is `22/22` passing. Python compilation, the exact nine-file canonical directory, all eight manifest-bound derived-file hashes, four bound source-code hashes, all summary checks and `locked_test_accessed=false` pass.
- Full-repository verification ran `933` tests: `930` passed and the same three pre-existing failures remained. They require the user-deleted `文档/项目说明` directory and the user-deleted `文档/研究进展/audit_tables.py` / `audit_table_numbers.py`; no R3 test failed and no new failure class appeared.
- The runner's non-empty-output safety gate was preserved. A verified candidate was copied into the canonical directory and rehashed. Cleanup of the redundant noncanonical candidate directory was blocked by execution policy, so `代码/artifacts/preflight/pi_jwm_r3_cpu_preflight_v1_reaudit_candidate/` remains only as a duplicate, not a documented formal result.
## 2026-08-04 R4 GPU数据索引与选模指标

- 正式训练样本入口为 `airfogsim_teacher_aligned_v3/window_index.csv`；字段包含 `decision_time`、输入/标签起止索引、`sample_id`、`seed` 和 `split`。GPU入口必须按该冻结索引或由冻结轨迹张量确定性派生窗口，禁止读取 `locked_test`。
- R2冻结选模分数由四项等权组成：信息边活动 `1-AUPRC`、活动链路速率 MAE（按训练尺度归一化）、任务生命周期 `1-macro-F1`、十项连续状态聚合归一化误差；任一公共项缺失时候选模块不得参与选择。
- `link.active_only_rate.mae` 只在真实活动且有效的信息边上计算；`task.lifecycle.macro_f1` 只在真实存在且有效的任务上计算；连续状态聚合覆盖物理节点、物理边、信息节点、信息边、信息流、任务和 DAG 状态。
- 验证集只负责 checkpoint/模块选择，校准集只负责阈值和分布校准；R4 不访问 locked-test。
- 现有 `r4_world_model.py` 已实现统一公共输出契约与候选专属辅助分布参数，`r4_objective.py` 在冻结 R3 目标之上显式追加 RSSM KL、异方差 NLL、hurdle 活动速率 NLL；GPU入口应复用这些模块，不另造模型接口。
- `r3_preflight_data.py` 已提供非锁定轨迹窗口加载、训练集统计归一化、信息链路活动标签冻结和显式批次构造；正式 GPU 入口应复用该语义并只增加批量拼接、训练循环和验证累计器。
- 旧 `formal_world_model_metrics_v1.py` 仍以旧物理边通信语义实现指标，不能直接用于当前 R4；GPU验证必须针对 v3 的 `information_edge_state` 和 `information_link_activity` 新写严格累计器，同时复用冻结 R2 指标定义。
- 当前 `make_explicit_batch` 一次只产生 batch=1；R4 GPU 需在不改变实体槽位、掩码和静态关系的前提下拼接多个 `ExplicitStateBatch`，并用微批次/梯度累积保持冻结的有效 batch size 32。
- v3 数据集共有 54 条已物化非锁定轨迹，每条 300 个时刻；36 条 train、12 条 validation、6 条 calibration，locked-test 只有完整性封条且没有目录。冻结 tensor contract 的原始窗口为 history 8、target 3。
- 十项选模连续指标和训练尺度已有 `evaluation_bundle_v3.py::_selection_scales` 生成逻辑，可直接复用尺度定义；模型预测处于训练归一化空间，验证累计时需按各字段 `scale` 还原为物理误差。
- 单条轨迹张量约 1.6--3 MB，54 条合计约 132 MiB；统一槽位为 44 物理/信息节点、1892 物理边、1892 信息边、588 流、481 任务。直接 batch=32 的图反向传播风险较高，应使用固定有效 batch 32 + 可记录的微批次梯度累积。
- 冻结 `window_index.csv` 每条轨迹约 290 个 8->3 窗口；R4正式入口需从这里确定性选取同一训练/验证窗口集并把窗口清单写入产物，不能让各候选随机看到不同样本。
- 远端真实张量 CUDA 冒烟：reference(hidden=16) 的 3 步前向/反向损失为 0.880306，耗时约 0.68 s；20 步开放环约 0.15 s，峰值分配显存约 100 MiB，所有显式预测有限。
- 未训练 reference 在全部 12 条 validation 轨迹、1/5/20 共 36 个窗口上四项公共指标均可计算，初始 `validation_protocol_score=4.49675`；该数值只证明评价链路闭合，不是正式结果。
- 首轮12臂运行曾在远端独立目录启动；其运行中状态已由下述统一源码第二轮正式结果取代，不能再作为当前状态或正式候选排名依据。

## 2026-08-04 R4正式GPU单模块筛选结论

- 统一源码第二轮完成12/12候选、0失败，总运行时间`3567.6608 s`。所有候选使用seed `20260803`、相同训练窗口、有效batch 32、最多30 epoch、patience 5以及validation的1/5/20步共36个窗口；calibration和locked-test均未参与架构或checkpoint选择。
- `graph_rssm_v1`以`4.482206671`获得最低验证综合分数；参考确定性Graph-GRU为`4.514384949`，前者相对降低`0.7128%`。检查点回读分数误差`1.21e-9`，低于冻结容差`1e-4`。
- 无跨图耦合臂为`4.531679555`，差于同预算参考臂；这为“保留显式跨图耦合”提供本轮经验支持，但单seed结果仍不能证明当前`CIP/CEP/CFL`门控参数化最优。
- 异方差头、ECC、soft presence、R-GCN和显式DAG消息臂与参考臂接近但没有在本轮公共综合分数上获胜；它们只能作为不确定性、DAG传播和动态拓扑等特定问题的R5组合/诊断候选，不能机械叠加为最终模型。
- 本轮绝对预测质量仍不够：胜者信息边活动AUPRC为`0.0015843`、任务生命周期macro-F1为`0.0147243`。R4只完成模块筛选与训练链路验证，尚未完成多seed收敛、策略器、论文baseline、真实闭环或最终效果验证。
- 第一轮运行因checkpoint复现容差过严及“活动链路必有正速率”的错误假设导致2臂失败。修复后从零重跑全部12臂，未混合两轮结果；活动但零实现速率现在保留在公共评价中，仅从hurdle正值速率辅助项中排除。
- 正式本地产物为`代码/artifacts/formal_training/pi_jwm_r4_gpu_screening_v1/`。远端和本地46/46清单哈希一致，12个checkpoint哈希和复现容差通过；R3 22项、R4 49项定向回归通过。全仓990项中987项通过，3项既有失败仍来自用户已删除的文档目录和审计脚本，与R4无关。

## 2026-08-05 R5 CPU准备初始审计

- R5不需要新建模型主干。`r4_world_model.py`已经统一Graph-GRU/Graph-RSSM、图编码、耦合、输出头、DAG和presence的公共输入输出；R5应只扩展配置合法性，使有限的已批准多模块组合可执行。
- `r4_objective.py`已经按配置附加RSSM KL、异方差NLL、hurdle项；A-E中C需要同时保留RSSM KL和异方差NLL，D/E需要保留RSSM KL并分别启用DAG/presence路径。
- `r4_checkpoint.py`已绑定组件、预算和上游溯源，可复用其严格加载机制，但R5正式预算和组合身份必须进入新的R5 envelope或等价严格绑定，不能把R4单模块checkpoint冒充R5结果。
- `r4_gpu_screening.py`已提供确定性非锁定窗口、batch拼接、设备移动和v3四项验证累计器；R5 CPU runner应复用这些函数并增加独立指标门、三seed协议和组合级诊断。
- R4分数的活动速率项量级约15，其他三项约1或更低，导致Graph-RSSM排名主要由速率MAE驱动。R5保留冻结综合分数用于复现，但必须并列检查四项公共指标和任务/结构基础门，禁止单分数自动定型。

## 2026-08-05 R5 CPU组合预检结论

- R4工厂的`elif`结构只能执行一个候选模块，不能靠改配置宣称多模块组合。R5采用白名单组合工厂：先构造参考/DAG/presence底座，再包裹Graph-RSSM，最后按C组合包裹异方差头；R4单模块行为保持不变并通过回归。
- 异方差包装器必须合并而不是覆盖下层RSSM的概率参数与执行元数据，否则C虽然前向可跑，RSSM KL证据会从输出契约消失。当前C同时保留`rssm_kl`和`heteroscedastic_nll`。
- 单个1步验证窗口可能没有活跃链路或有效数据流，因此AUPRC、active-rate MAE与连续状态汇总按协议应为`not_computable`。CPU门使用1/5/20步最低充分窗口集合；禁止把无样本指标填0。
- A-E×3 seed正式CPU门15/15通过；参数量分别为A 3850、B 4576、C 5234、D 4736、E 4576（CPU预检hidden=4，仅用于执行成本检查）。这些数字和一次更新后的指标不能用于模型排名。

## 2026-08-06 R5多seed正式结果

## 2026-08-07 R5.1正式训练完成后的初始证据

- B/F/G/H/J共15份同协议证据已经齐全；新训练F/G/H/J各3 seed，B的3 seed来自逐checkpoint验真的同协议复用结果。
- 冻结综合分数均值（越低越好）：J `4.231782`、B `4.450672`、G `4.454345`、F `4.468286`、H `4.508760`。
- J并非全面胜者：它的物理距离、DAG未完成父任务数、deadline和队列误差明显较低，但信息边速率RMSE为`42.3114`，明显差于B的`30.1467`和G的`29.8932`；其AUPRC均值又被单seed `0.0622`拉高，必须报告seed方差与分horizon证据。
- G的任务生命周期Macro-F1均值为`0.20198`，高于B的`0.12856`，但物理距离、DAG和deadline总体更差；H没有综合优势，F保留为无耦合消融。
- 因此步骤3不能简单按综合分数选择J。合理的冻结对象是“R6工作候选集合及角色”，而不是R9最终方法；需要先完成分horizon和配对门。
- 现有可复用实现包括`r5_analysis.py`/`analyze_r5_multi_seed.py`以及R5.1正式bundle验证器；应扩展现有统计接口，不能再造独立且口径不一致的分析框架。
- `r5_analysis.py`已经提供严格矩阵校验、方向感知的同seed benefit、精确双侧sign-flip检验、t区间、收敛/成本统计和自校验manifest写出；R5.1应复用这些基础函数，仅增加B/F/G/H/J矩阵装配、任意baseline配对、分horizon记录和R6角色门。
- `R4ValidationAccumulator`会按传入窗口累计14项离线指标，但正式run report把1/5/20窗口合并后才写出；因此分horizon不能从聚合报告反推，必须用已验证checkpoint在冻结validation/calibration窗口上分别重放horizon 1/5/20，且仍不得访问locked-test。
- 正式bundle的窗口JSON保留的是远端绝对`tensor_path`，本地不可直接照路径读取；分析器必须从本地冻结dataset root确定性重建窗口，并逐项比较`trajectory_id/split/history_start/history_end/target_start/target_end/horizon_steps/environment_seed`，不能只按seed重新随机抽样后默认相同。
- F/G/H/J checkpoint使用`load_confirmation_checkpoint`，B复用checkpoint仍在原`pi_jwm_r5_gpu_training_v1`并使用`load_r5_checkpoint`；分horizon分析必须分别走两个严格loader，再统一进入同一个`R4ValidationAccumulator`，不能复制或转换checkpoint。
- 本地冻结dataset、R2 evaluation bundle和原R5训练根目录均存在。正式训练代码已经提供`_validate_candidate`、`load_selection_scales`、normalization stats和严格loader所需bindings；本轮应以薄CLI编排这些现有接口，而不是修改训练器。
- 每个seed的validation窗口本身已按1/5/20分组且每个horizon覆盖12条validation轨迹；calibration每个horizon覆盖6条轨迹。分horizon比较必须在相同training seed下使用其原冻结窗口清单。
- 训练器现有`_window_identity`恰好比较环境seed、历史/目标边界、horizon和split而忽略机器相关绝对路径，可直接用于本地窗口身份验收。
- 现有`_validate_candidate`包含`torch.cuda.manual_seed_all`与`torch.cuda.synchronize(device)`，不能原样用于CPU。新分析模块应复用`R4ValidationAccumulator`和批构造逻辑，提供只读、`model.eval()`、`torch.no_grad()`、CPU安全的同口径重放；不修改正式GPU训练函数。
- 为避免事后按结果挑规则，R6角色门在分horizon运行前固定：B始终是强制控制；整体候选必须三seed综合分数同向优于B且信息速率/活动速率无超过2%的均值退化、任务Macro-F1无超过0.02绝对退化；专长候选要求预声明指标族均值改善至少10%且至少2/3 seed同向。未过门者只保留为消融证据。

## 2026-08-07 R5.1最终本地分析发现

- 本地CPU严格重放可复现GPU报告：15个checkpoint最大综合分差`1.2208e-05`，说明分horizon数据不是由聚合报告拆分或另采样产生。
- B/F/G/H/J三seed综合分均值分别为`4.450672/4.468286/4.454345/4.508760/4.231782`。J虽然3/3 seed改善综合分，但信息边速率RMSE由B的`30.146708`升到`42.311381`，3/3退化约40.28%；任务Macro-F1也由`0.128558`降到`0.089510`。
- G的任务Macro-F1均值为`0.201977`，较B绝对提高`0.073420`且2/3 seed改善，但综合分仅1/3 seed改善、连续状态误差0/3改善。因此G只能作为任务生命周期专长候选。
- J的连续状态归一化误差由B的`0.676524`降到`0.422650`，均值改善37.50%且3/3 seed改善；该优势在1/5/20步均保持。但J任务Macro-F1由1步`0.236598`降至20步`0.055235`，证明其长时域任务语义仍有缺陷。
- H的信息边速率RMSE 3/3优于B，但综合分0/3改善；F的任务Macro-F1 0/3改善。两者均不能进入总体候选，其中F只保留去耦消融，H退出当前R6主集合。
- 没有任何挑战者通过全部总体门，所以B是R6的均衡主工作候选；这不是最终方法定型。R6应同时保留G/J专项分支用于验证“任务语义”和“连续状态”能否在策略闭环中产生真实收益。

## 2026-08-07 R6 CPU预检发现

- 正式数据已经足以支撑R6入口审计：54条非锁定轨迹全部具有双图、CPU动作、资源与真实系统结果账本；23,255条CPU动作及432个轨迹文件的hash均可验证，硬约束违规为0。
- 三种现有CPU策略在全数据中各18条，在validation中各4条，但它们不是同一状态/同一seed下的配对反事实，因此均值只能作为量纲检查和闭环下界，不能用于因果策略选优。
- validation观测中，feasible exploration完成率/吞吐量较高但总能耗也较高；equal share的P95/P99时延和Jain公平性较好。该权衡说明R6必须保留多目标分项指标，不能压成单一reward后只报总分。
- `dependency_data_delivery_rate`在当前DAG先后约束数据上是`not_applicable`；`action_regret`、`uncertainty_coverage`和`ood_transfer_score`仍是`not_computable`。R6/R7/R8只能在生成相应策略对照、分布输出或迁移证据后改变状态，不能用0占位。
- R6下一合法步骤是冻结动作空间、mask、安全投影、reward和同状态配对执行协议，并跑规则/局部搜索CPU闭环；Actor–Critic/PPO GPU训练仍未获入口门通过，世界模型保持stop-gradient。

## 2026-08-07 R6 CPU配对smoke发现

- 配对runner只改变CPU回调，卸载、RB、移动、信道、任务到达和DAG先后均沿用正式AirFogSim runtime；同一场景、seed和配置指纹下四个策略臂可成组运行。
- `base`环境缺少`osmnx`导致首次smoke失败，说明不能假设当前Python环境等同于数据生成环境；切换已有`airfogsim`环境后smoke恢复，未安装新依赖。
- 1个validation配对的4/4臂动作合法率为100%，CPU/RB/任务流/能量硬约束为0；该结果仅证明执行链，不作策略性能结论。

- A-E×3 seed共15/15完成，训练、恢复、checkpoint复现和本地manifest验收均通过；结果完整性不再是当前阻塞项。
- B/C的validation综合分数为`4.4507/4.4508`，低于A的`4.4958`，同seed均3/3改善约1%；calibration方向一致，但三对样本的双侧精确检验最小只能为`p=0.25`。
- 综合分数不能单独定型：A的链路活动AUPRC为`0.02253`，B/C仅约`0.00201`；B/C只稳定改善活动链路速率MAE，连续状态和任务Macro-F1均存在seed不一致。
- C相对B的综合变化接近0，当前报告又缺少NLL、覆盖率和区间宽度，不能确认异方差头价值。D相对B改善连续状态约1.02%，但0/3改善速率和综合分数；E无稳定收益。
- 当前方法判断为：B是工作候选而非winner，A必须保留，C/D仅作专项诊断，E停止主路径。R2全指标复评完成前不得进入R6或固定最终世界模型。

## 2026-08-08 R6 CPU正式配对闭环发现

- 正式runner在项目既有`airfogsim`环境完成54个非锁定base spec与四个CPU策略臂的同场景配对，共216/216 runs；三类split为`train=144`、`validation=48`、`calibration=24`，没有读取locked-test，也没有更新世界模型或启动GPU。
- 54个pair group均完整，所有策略臂共享`scenario_id`、环境seed、配置指纹、最大仿真时间和协议版本；`action_legal_rate_min=1.0`，硬约束违规0，失败run 0。8个bundle文件独立SHA-256复算无差异。
- 真实闭环指标状态为`available=3888`、`not_applicable=216`、`not_computable=648`。缺失指标继续保留三态语义，没有以0替代；`action_regret`仍因尚未冻结可执行效用和反事实最优动作而不可计算。
- validation配对差值只作为下一阶段诊断输入：`local_search`相对`equal_share`的任务完成率和吞吐量均值分别约增加0.0063和0.0384，但总能耗约增加11.14；`deadline_aware`和`feasible_exploration`的时延/能耗也存在不同方向的权衡。因此当前不能宣布某个规则策略获胜，仍需在同一协议下加入学习策略，并按事前冻结的分项指标和配对统计比较。
- `base`环境缺少`osmnx`导致的smoke失败已保留并记录；切换到项目已有`airfogsim`环境后正式运行成功，未修改第三方依赖。外层工具等待上限先返回但子进程继续完成，最终只验收一次正式bundle，未重复启动造成重复样本。

## 2026-08-08 R6学习策略CPU预检发现

- 冻结候选B可以直接为独立策略器提供24维显式状态和32维Graph-RSSM belief；策略输入只使用历史观测、历史动作和当前mask，未来target未泄漏，策略梯度也没有回传或修改世界模型。
- 真实CPU动作约束不是一个batch级总容量，而是44个信息代理节点槽位上的逐节点容量和481个任务槽位的任务—节点映射；本次validation状态有435个活动任务分布在38个活动节点上。安全投影必须按节点分组，不能只对全部任务做一次总量缩放。
- 按比例缩放在`float32`中可能留下约`1e-7`量级的正残差。本次真实状态捕获到两个节点分别超量`7.15e-7`和`2.38e-7`；因此实现增加缩放后残差回收，而不是放宽硬约束阈值。
- Actor–Critic/PPO的一次CPU更新只证明动作分布、log-prob、value、ratio、entropy和梯度链可执行。常数advantage/return不能支撑策略性能或收敛结论，也不能代替正式reward和rollout协议。
- 卸载/RB的通用mask接口已经实现，但本轮正式CPU门将它们约束为安全no-op，因为真实RB可用域和联合动作执行协议尚未冻结。GPU阶段若学习卸载/RB，必须从正式环境构造真实mask和结果，不能把no-op占位当成训练数据。
- R6学习策略CPU入口已经Ready；GPU策略训练仍需冻结多目标reward/约束处理、GAE或rollout样本生成、训练seed与预算、checkpoint/失败保留以及与四个CPU策略臂的同场景配对评价矩阵。

## 2026-08-08 R6在线闭环修正发现

- 真实reward并不等于真实状态闭环：旧runner在候选动作偏离教师轨迹后继续读取冻结张量，会形成“真实反馈＋错误状态”的训练样本；该路径已被在线重采集替代。
- 冻结轨迹不能提供五类非默认候选的反事实结果标签；候选覆盖仅1/6，因此reward surrogate不可辨识，不能通过复制默认reward或规则估计补标签。
- 在线图必须保留最近8步出现过的节点/边词表，并用presence表示对象离开；只保留当前节点会使历史通信边成为悬空边。网络附着但无通信边的cloud快照还必须按正式图边界同步过滤。
- 联合动作日志必须写入最终执行动作；CPU分配器只能在AirFogSim执行时调用一次，否则`feasible_exploration`的RNG会被提前推进。
- 32步CPU实测约0.242秒/环境步；单run 100k粗估6.73小时。正式矩阵应先2k GPU smoke，再10k×18阶段，确认资源与学习曲线后原子续训到100k，不能直接盲跑完整预算。

## 2026-08-13 P2单步非训练集成发现

- AirFogSim的本地目标会立即进入计算，而远端目标先通信；但“本地/远端”同时改变目标和RB适用性，不是严格单因素候选对，已从正式夹具中淘汰。
- 正式夹具使用两个独立同seed环境，分叉前环境、Python RNG和NumPy RNG哈希一致；两候选分配完全相同的50条`(t,f,e,RB)` COO，仅改变远端目标`RSU_2/RSU_0`。
- 两条候选都把`Task_4`的真实剩余数据`0.3350039866`在同槽送达；逐RB有效速率和outage来自AirFogSim直接矩阵，不由聚合rate反推。
- 两条候选均在目标节点进入CPU callback；`computed_after-before`与`PIJWM-CPU-Inner-Rule-v1`的served work一致。
- AirFogSim原生通信能耗输入存在发送键记账错误；P2在能耗更新前依据direct transfer event重建source/target总量，并逐UAV验证能耗方程。artifact明确标注该修复，不声称原生实现正确。
- 首帧三个E1历史结果字段使用`NO_HISTORY`和零填充但`valid_mask=false`；当前逐RB attenuation/SINR/interference/rate/outage仅在直接同RB来源存在时有效，不补造13维旧槽。
- 单步证据只证明真实动作到通信、CPU和能耗的一步链与v4字段边界；不证明多步词表稳定、E1上一槽回填、正式v4数据集、模型训练或候选世界模型rollout规划器。
# 2026-08-13 P2 多步前置时序复核

- 发现关键口径冲突：`run_p2_single_step_collector_preflight_v1.py::_field_audit` 将 transfer event 中的 `attenuation_db` 写入 `pre_link.channel_attenuation_*`，并声明来源为 `direct_current_per_rb_attenuation`。
- 真实执行顺序为：候选动作 setter -> `AirFogSimEnv.step()` -> `_updateTraffics()` -> `_updateWirelessCommunication()` -> `_compute_communication_rate()` -> `updateFastFading()` -> `computeRate()` -> `_event_from_profile()`。因此该 event 衰减是在动作 setter 之后、当前槽交通与快衰落更新之后采集的 outcome-side channel snapshot，不是 action-pre observation。
- 影响边界：P2 单步真实 offload/RB/communication/CPU/energy 闭环事实不因此失效；但正式 bundle 中 `pre_link.channel_attenuation_mean_db/std_db` 与 `pre_rb_optional.channel_attenuation_db` 的 action-pre 证据口径不成立，必须先修订/重放，随后才能扩展多步 E1 回填。
- 推荐修订：在任何候选动作 setter 前读取 decision-time CSI，单独写入 action-pre channel observation；保留 `computeRate()` 后捕获值，但明确标为 outcome-side channel snapshot，禁止送入同槽决策输入。
# 2026-08-13 P2 时序修订实施发现

- setter 前观测边界可以在 `validate_candidate_action` 成功之后、CPU callback 安装和 AirFogSim offload/RB setter 之前，以一次显式 callback 固定；非法动作测试证明该 callback 和所有 setter 均未触发。
- 时序正确性不能依赖“动作前后 CSI 数值不同”；确定性证据应由调用 trace、`capture_phase`、端点/RB 身份和来源方法共同组成。
- 单步 writer 与 verify-only 必须调用同一纯时序校验器；否则 `validation_report.passed=true` 或 manifest artifact 哈希只能证明文件未变，不能证明字段来源语义正确。
- 修正版单步正式bundle中，两候选各有50个setter前`channel_manager.getCSI`衰减值；动作后事件只保留`outcome_channel_attenuation_db`并标记`outcome_only_not_same_frame_decision_input`，旧模糊`attenuation_db`键已移除。旧真实通信/CPU/能耗结论保留，旧action-pre字段证明撤回。
- 多步正式fixture在同一seed 0环境真实执行3步：第0帧1个offload和50条RB COO，实际交付`0.33500398659608244`；第1/2帧为空动作且真实step，第1帧零通信；第2帧将该零结果回填为`valid=true/missing_reason=none`。这证明合法零值与缺失可区分。
- node/edge/flow词表已实现append-only与事务验证；本fixture只出现1条已执行无线边，因此只能证明“已观测通信边身份稳定”，不能证明完整E0物理图/信息图、动态presence或CEP已实现。
- PowerShell 5.1读取UTF-8 manifest时若不显式指定`-Encoding UTF8`，中文`代码/`路径会显示为`浠ｇ爜/`并造成假的source缺失。显式UTF-8后，单步8 artifact/21 source和多步7 artifact/26 source独立复算均为0不匹配。
- 合并前干净worktree集成预检暴露出分支不自包含：P2已提交代码依赖4个仍未跟踪的PI-JWM模块，且manifest未完整绑定CPU适配层及相关测试。根因是Git提交边界和运行/证据依赖边界不一致；已用两条先失败后通过的闭包测试补门，并将必要模块与测试纳入分支。AirFogSim是独立第三方checkout，不纳入主仓库提交。
- 第一层闭包补齐后，干净worktree的77项测试仍有1个错误：runner通过`sys.path`动态导入未跟踪的`small_experiments/airfogsim_strict_dual_graph_preflight.py`。已将该现有helper及其16项测试纳入Git和两套manifest；修正后干净检出不再依赖主工作区的隐含未跟踪PI-JWM文件。
- 合并后的干净`main`上，P2聚合77项和动态helper 16项均通过。全仓`unittest discover`运行933项但有283个错误、2项跳过；已观察到的根因包括环境缺少`sklearn`、Python 3.10没有`tomllib`，以及历史正式数据脚本依赖仍未跟踪的`pi_jwm.airfogsim_dataset_v2`。这不推翻P2范围验证，但阻止“全仓测试通过”的表述。

# 2026-08-13 P2正式v4全双图采集器设计核查

- AirFogSim环境直接维护`vehicles/UAVs/RSUs/cloudServers`及移除列表，且每步先更新交通再执行任务、无线、有线、计算、存储和能耗；因此动态节点presence必须按明确的动作前快照时点采集，不能从动作后的传输事件反推。
- AirFogSim提供任务DAG、卸载路由、无线激活链路和RSU/Cloud有线通信路径接口；这些来源可以支持物理节点/边、信息任务/DAG边和执行映射的分层采集，但不能自动证明所有理论CEP关系都有直接语义载荷证据。
- 一次并行`rg`核查中至少一个无匹配搜索返回退出码1，组合工具把整组标为失败；按AGENTS规则记录为“未找到结果”，后续拆分读取，不误判为权限或工具故障。
- 历史`formal_airfogsim_graph_v1.py`明确把任务DAG限制为precedence-only并禁止合成dependency-data flow，适合作为任务生命周期/DAG状态参考，不足以证明v4完整通信信息边或CEP。
- 历史strict preflight分别维护物理图、任务信息图、MN/ME/EP关系；其中EP只有在信息边具有直接`data_mb`和直接传输路径时才通过，未建模的DAG依赖不能用路由或后处理补造。该严格拒绝原则应保留。
- 当前v4字段设计中的`structure.endpoint_index`与`structure.cep_physical_edge_index`针对“具有同端点通信语义的信息代理边”；任务DAG precedence边的端点是任务，不天然对应同端点物理边。正式采集器必须把通信流边与任务DAG边分开建模，或明确收缩理论，不能共用名称伪造CEP。
- AirFogSim `ChannelManagerCP`为V/U/I的V2V、V2U、V2I、U2V、U2U、U2I、I2V、I2U、I2I端点组合维护全CSI矩阵，`activateLink`只标识本槽动作实际使用的端点/RB；源码没有一个可直接当作统一无线“可通信范围真值”的门。
- 因而物理边至少需要两个正交状态：`edge_present`表示端点当前存在且AirFogSim对该有向类型定义信道/有线机制，`edge_active`表示本槽确有调度或直接传输事件。不能用`edge_active=false`推断物理边不存在，也不能自行用距离阈值删边。
- 动态节点离场/重现可由环境当前V/U/I/C集合直接观测；边presence应由稳定append-only端点词表与当前节点presence派生。无线全端点对可能是O(N^2)，正式采集器需报告每帧候选边规模和序列化成本，但不能为省算力静默改成仅激活边。
- AirFogSim同一`Task`对象会依次经历offload、compute、return；`_transmitted_size`在完成一跳后归零，`_to_offload_route`逐跳删除，且`changeOffloadTo`可改写未完成路由。因此`task_id`不能单独充当通信flow身份，否则卸载载荷、返回载荷和路由修订会混在一起。
- 推荐逻辑flow身份为`(trajectory_id, task_id, phase, route_revision)`，其中`phase in {offload, return}`，`route_revision`只在直接调度动作创建/修改路由时递增。每个实际承载跳另建`communication_flow_edge=(flow_id, hop_index, src, dst)`，CEP只作用于该跳与同端点物理边；权威COO仍是`(time, flow, communication_flow_edge, RB)`。
- 局部/无需传输的任务动作可以有offload action记录，但不创建伪通信跳或RB分配；其通信flow状态明确为空/不活动，而不是自环物理边。

## 2026-08-13 RB与任务动作执行语义复核

- `CommunicationScheduler.setCommunicationWithRB`只将`task_id -> RB列表`写入环境，并对RB编号做`% n_RB`。正式采集器必须在setter前拒绝越界RB，不能把第三方静默取模后的结果当作合法动作证据。
- `AirFogSimEnv._allocate_communication_RBs`从当前任务的`getToOffloadRoute()[0]`取得真实承载首跳；COO的`information_edge_index`不会驱动AirFogSim执行。因此正式协议必须显式验证`flow/hop/src/dst/physical edge`与当前路由首跳一致，否则拒绝。
- `ChannelManagerCP.activateLink`不会禁止不同链路复用同一RB；`computeRate`会把同RB发射功率累加进干扰矩阵，再计算SINR、outage和rate。因此RB复用是AirFogSim真实可表达的现象，不应被采集器定义为全局非法；但同一COO四元组重复、同一任务重复RB、端点与路由不一致仍是非法。
- 有线通信按`wired_manager.hasLink(tx, rx)`与当前路由执行，不经过RB；正式COO不得为有线flow伪造RB。
- waiting-to-return任务通过`task_return_routes`在`_updateTask`中启动回传，回传必须使用独立`phase=return`与`route_revision`，不能复用卸载flow身份。

## 2026-08-14 P2-C source-closure recovery findings

- 本机微信归档`AirFogSim_clean_runnable.zip`提供P2-B所需的全部83个相对路径，其中82个SHA-256与manifest完全一致。
- 唯一差异是`airfogsim/manager/energy_manager.py`：归档哈希为`074479985b93939c4d85aa11e4f6db2bcd8c132f57c7ca03bf459774df88479c`，manifest期望`da2599f33af0dd38db09affdbf1eeb94f39a2b140cc6fb596c49f4f462cd3b90`；LF/CRLF和末尾换行变体均不能解释差异。
- 本机Codex会话中只有2026-05-05与2026-08-11两个会话提到`energy_manager.py`。8月会话当前命中为依赖枚举/搜索输出，没有发现该文件正文、补丁或期望哈希的历史副本。
- 2026-05-05文件是审批审计会话，只嵌入了原任务的截断增量；它指向真正的归档会话`019dc899-2747-7cd2-911c-2184d0474ffc`。该原会话已在`C:\Users\Lenovo\.codex\archived_sessions\`及一次只读备份中定位，后续应直接检索原会话而不是继续解析审批包装层。
- 原会话证明AirFogSim于2026-05-05从`ZhiweiWei-NAMI/AirFogSim`直接克隆。后续列表记录中的`energy_manager.py`长度为3478字节，与当前微信归档候选一致；会话内没有发现针对该文件的`apply_patch`、正文读取或直接写入记录。
- 这支持“微信归档接近最初clone”的判断，但不能证明manifest期望版本；目标哈希更可能来自后续未记录工作树、Git对象或另一归档，仍需按字节恢复。
- 进一步搜索全部归档会话后，定位到2026-07-13对第三方`energy_manager.py`的两次真实`apply_patch`：第一次加入逐分量能耗快照逻辑，随后第二次撤回这些逻辑并改由PI-JWM侧读取私有状态。会话明确记录两次补丁均成功。
- 因此当前单文件哈希差异有了可检验假设：逻辑虽然撤回，但两次补丁可能把原始全CRLF文件变成局部/整体不同的字节换行布局。此前只检查“全LF/全CRLF”不足以排除混合换行。
- 下一步不是凭该假设直接修改项目，而是在临时副本重放两次完整历史补丁，并用manifest目标SHA-256作唯一判据。
- 历史补丁重放已精确命中目标：重放文件长度3457字节，SHA-256为`da2599f33af0dd38db09affdbf1eeb94f39a2b140cc6fb596c49f4f462cd3b90`；原归档文件为3478字节和`074479...`。
- 两者在统一换行后文本逐字符相同。字节差异恰为原文件65个CRLF，经历史补丁往返后变成44个CRLF和21个裸LF；这证明manifest绑定的是历史补丁留下的混合换行字节版本，而不是不同逻辑实现。
- 该证据允许把精确重放文件与归档的其余82项组成临时83项候选，但在83/83整体复算通过前仍不写入项目引用目录。
- 临时候选按P2-B manifest逐项复算为83/83；项目目标随后确认是普通空目录而非junction，并恢复同一候选。目标目录再次独立复算仍为83/83、0 mismatch。
- 这关闭了“文件字节是否可恢复”的子问题，但完整`source closure`仍需P2-B/P2-C verify-only和focused suite复跑后才能正式判定。
- P2-B verify-only已通过；P2-C首次因缺少脚本强制参数退出，补齐canonical `--bundle`和`--output-dir`后通过。
- 第一次恢复后focused suite运行151项，原16个`FileNotFoundError`全部消失，但5项在`channel_manager_cp.py`导入时打印emoji并因GBK stdout触发`UnicodeEncodeError`。这是可复现的运行编码边界，不是文件哈希、模型逻辑或locked-test失败。
- 不应修改manifest绑定的第三方源码来规避该错误；最小假设检验是仅设置`PYTHONUTF8=1`重跑完全相同的151项。
- `PYTHONUTF8=1`复跑通过，并实际展开158项全部通过；此前151项是因为observer类`setUpClass`失败导致7项未执行，不能将首轮称为151项中的仅5项失败而忽略未展开项。
- 当前canonical P2-C JSON仍是恢复前快照，`blocking_reasons`仍含`input_manifest_source_closure_failed`；其余四项为action rejection分母、formal scale、formal split和scenario matrix。应先生成新候选审计，确认只移除source-closure阻断，再更新canonical。
- 新候选审计已生成并通过自身verify-only。逐文件diff证明候选只发生三类预期变化：移除`input_manifest_source_closure_failed`、把83项`missing_sources`清空并将输入manifest检查改为`passed=true`、更新审计JSON的artifact哈希；formal config逐字节不变，其余四个阻断完整保留。
- 提升后的canonical再次通过P2-B/P2-C verify-only、83/83复算和158/158 focused tests。
- 但随后核查发现P2-C runner当前只绑定审计代码、脚本、两项测试、实施计划和P2-B设计，未绑定本次P2-C研究进展文档；因此该文档修改后verify-only仍通过。历史会话中曾存在补路径补丁，但当前`main`没有该行，不能把历史补丁当作当前实现。
- 最小修复已按RED/GREEN完成：P2-C文档进入现有`CANONICAL_SOURCE_PATHS`，manifest包含portable key，临时副本文档篡改会触发`source hash mismatch`；没有新建第二套manifest，也没有改变审计算法或四个剩余阻断。
- 提升后的canonical通过双`--verify-only`、83/83依赖哈希和159/159 focused suite；P2-C仍因attempt/reject分母、scenario matrix、formal scale和formal split四项未冻结而阻断，不能生成正式轨迹。

## 正式采集器coverage policy比较

1. `disjoint_only`：无线flow使用互不重叠RB block。归因简单，但覆盖不到真实同RB干扰，不能作为唯一正式策略。
2. `reuse_only`：不同承载边允许复用RB。可覆盖干扰，但若所有样本都采用会造成偏置，不利于定位基础链路问题。
3. `balanced_two_arm`（推荐）：常规帧采用确定性正交分配，干扰覆盖帧允许不同物理承载边复用RB；两类都来自同一自然任务、节点运动、信道和CPU规则，并记录`resource_policy`。这是数据覆盖策略，不改变模拟器合法动作域。

## 推荐帧级执行规则

- action-pre先快照节点presence、物理结构边、任务状态、DAG和当前路由，再枚举`waiting_to_offload/offloading/waiting_to_return/returning`任务；computing与本地执行任务不伪造通信动作。
- `waiting_to_offload`只在目标、route和每一跳结构边均有直接证据时调用offload setter；否则任务保留在等待态并记录`not_selected`原因。
- 已在`offloading/returning`的任务不得重复调用offload setter，只对当前首跳分配无线RB或记录有线承载；`waiting_to_return`必须先设置并验证return route。
- setter前验证task状态、route首跳、节点presence、`edge_present`、channel type、RB范围、COO唯一性及flow/hop/edge对应关系；验证失败时不执行`env.step()`。
- 每个真实`env.step()`记录task lifecycle、DAG transition、logical flow、communication-flow hop、CEP、edge active、RB、rate/outage、CPU和energy；action-pre与outcome严格分离。

- 2026-08-19 v2 collector adapter 已通过 5 秒真实 formal smoke：`formal_collector_ready=true`，双图/resource checks 全部通过，9 种物理方向齐全；该 smoke 未生成正式数据集。
- v2 CPU callback 可能在同一 simulator time 被多次调用；adapter 为每个同一时间/节点 callback 保留直接执行顺序的纳秒级唯一时间戳，保留真实 `allocated_cpu*slot_seconds` 计算并使容量审计按 callback 边界验证，未合并或伪造工作量。
- 能量行现由 energy manager 直接状态、同槽 transfer rows 和 energy config 成本组成，`energy_equation_valid` 与 `channel_energy_input_valid` 在真实 5 秒 smoke 均通过。
- 全量 `airfogsim` 环境回归共 1370 项；主体断言通过，但 275 项因该环境缺少 `tomllib`/`scikit-learn` 导入失败，另有脚本型测试因未提供其 CLI 参数退出。不能把该结果表述为全仓全绿；核心变更使用定向测试验收。
- 适配器复核发现 energy ledger 循环曾错误嵌套在 CPU policy callback 循环内；已改为逐帧收集 energy rows、再统一生成 CPU ledger，避免重复写入和外层 `frame` 残留引用。
- 真实高负载/高密度 5 秒 smoke 暴露 AirFogSim 的两个字节语义：transfer `delivered_data` 为任务完成时的守恒截断值，而能耗边界使用每个 profile 的完整 `rate * simulation_interval`。energy input 已改用现有 `_wireless_totals()` 的 `planned_capacity` 汇总，保留 transfer ledger 的截断进度，不用代理字段掩盖差异。
- 修复后真实 smoke 通过：50 帧、50 条 action attempts、`validate_attempt_records` 0 错误、`formal_collector_ready=true`、`PIJWM-AirFogSim-Full-Collector-v2`、dual-graph/resource gates 全部通过、能耗方程和 channel input 均有效、9 类物理方向齐全；仍未生成正式 60 条轨迹。
- 新增 `formal_airfogsim_protocol_audit_v1.py`，独立重算用户冻结协议与代码 specs；实测通过场景矩阵、60 条规模、`36/12/6/6` split、seed 公式、每场景 5/5 资源臂、固定运行参数、校准 report/probe SHA-256 和 locked-test 封存门。该 audit 只解除协议门，仍保持 `formal_data_approved=false`。

## 2026-08-19 正式候选数据独立验收

- 正式 builder 已生成 60 条轨迹，六场景各 10 条，split 为 `train=36`、`validation=12`、`calibration=6`、`locked_test=6`；两类资源臂全局各 30 条、每场景各 5 条，seed 唯一。
- 独立审计 `formal_airfogsim_data_audit_v1.py` 不直接采用 builder 汇总作为唯一证据：逐轨迹重算 manifest 文件哈希，读取并验证 `action_attempts.jsonl` 的 18,000 条记录，重新执行 `validate_dual_graph_v2_bundle`，并核对资源/能耗验证报告、运行时 contract、9 类物理方向和顶层索引。
- 验收结果：`audit_ready=true`、`formal_data_approved=true`、`training_eligible=false`、`locked_test_accessed=false`；metrics 长表实际是多指标行并以 `seed` 为主键，独立审计已按真实 schema 验证 54 个非 locked seed，未将 locked-test 纳入 metrics。
- 本阶段只批准“正式候选数据可作为后续训练输入的来源”，没有批准训练、调参、GPU 或最终方法；`formal_training_ready=false` 仍由 tensor contract、training statistics 和 locked-test 封存阻断。
- 审计脚本两次初始失败均被定位为审计逻辑与实际 CSV schema 不一致，修正后第三次深度审计通过；失败证据保留在 `code/artifacts/audit/pi_jwm_p2c_formal_data_audit_20260819_failed*`，canonical 目录为无后缀版本。

## 2026-08-19 训练前 Tensor 化首轮诊断

- 首次运行 `build_formal_airfogsim_tensor_v1.py` 处理正式 train seed `10000` 时失败：`ValueError: duplicate snapshot (0.0, 'RSU_0')`，失败发生在 `airfogsim_tensor_v2._group_unique_rows`，不是环境依赖或权限问题。
- 对真实 `dual_graph_v2_bundle.json` 独立统计确认：`source_physical_node_snapshots` 15139 行只有 5078 个 `(observed_time,id)` 唯一键，5033 个键重复；重复行内容相同。adapter 当前把每帧 decision、execution、outcome 三个 phase 都写入同一物理状态序列，而 execution/outcome 同一时间，下一帧 decision 又落在该时间。
- 现有 tensor contract 的状态数组按单一 observed-time 网格建模，动作后差异已有 transfer/CPU/energy/outcome ledger 承载；因此不能通过任意覆盖或去重掩盖重复。修复假设是物理 node/edge time-series 只取决策时刻 snapshot，保持 outcome 侧 ledger 独立。
- 尚未修改生产代码；下一步先把该行为写成 RED 测试，再实现最小 adapter 修复并重新生成本地 tensor 证据。
- RED 回归 `test_physical_state_stream_uses_one_decision_snapshot_per_time` 按预期从 3 条 snapshot 失败；最小修复将 execution/outcome 调用标记为 `include_physical=False`，仍保留 outcome task/DAG rows 和 transfer/resource ledger。adapter 2/2、full collector v2 6/6、formal tensor fixture 1/1 已通过。
- 旧 v2 formal data 的 manifest 是修复前 adapter 生成的，不能直接视为新代码的数据证据；下一步使用修复后 adapter 生成新的 `pi_jwm_v4_formal_candidate_v3`，旧 v2 目录保留用于历史追溯。
# 2026-08-15 接续文档核验发现

- 主工作树 `main` 当前为 `7d85833`，存在大量用户/历史未提交改动，不能清理或覆盖。
- P2 Attempt/Reject Ledger v1 在隔离 worktree `.worktrees/p2-action-attempt-ledger-v1`、分支 `codex/p2-action-attempt-ledger-v1`，HEAD 为 `abfbe10`，证据文档还有未提交修订。
- 另有 `p2-c-scale-distribution-audit` 与 `sparse-event-diagnostic-v2` 两个隔离 worktree，接续文档必须区分其历史角色和当前主执行分支。
- 新接续文档不能把隔离分支成果写成已合并主线，也不能把 P2-C 的候选审计写成正式数据批准。
- P2-B v2候选的机器报告确认：6条自然episode、120帧；`natural_reference` 120次attempt全部accepted，拒绝率0.0；全角色合计260次attempt；`training_eligible=false`、`v4_dataset_complete=false`、`candidate_rollout_planner_complete=false`。
- P2-C v2当前只有 `pi_jwm_p2c_scale_distribution_audit_v2_pre_document_closure_20260814`，真实受管文件为 `p2c_scale_distribution_audit_v2.json`、`p2c_formal_data_config_candidate_v2.json` 与 `manifest.json`；最终文档绑定候选目录尚不存在。
- P2-C v1文档列出的四项阻断中，ledger v2已为拒绝率建立真实分母；剩余正式scenario matrix、规模、split三项仍未冻结。
- P2-C v2机器报告状态为`blocked`，阻断项精确为`formal_scale_not_frozen`、`formal_split_not_frozen`、`scenario_matrix_not_frozen`；`formal_data_approved=false`。
- P2-C v2从`action_attempts.jsonl`独立重算natural-reference为120 attempts、120 accepted、0 rejected、0 quarantined，frame/replay alignment与schema/transition门均通过。
- 外部知识库`PIJWM主文档.md`是固定理论/方法边界；`8.12之后推进.md`是动态进度，但截至2026-08-14仍写四个P2-C阻断，尚未同步ledger v2关闭拒绝率分母门的分支事实。
- 主文档明确：信息特征数量不是目标；更少但可靠的信息若效果相当且开销更低，应优先。世界模型规划器必须逐候选实际rollout，只读belief直接打分只能称direct policy对照。
# Findings

## 2026-08-19 训练前 tensor 结构通过、语义特征门阻断

- 修复后正式候选目录 `code/artifacts/formal_data/pi_jwm_v4_formal_candidate_v3/` 已通过 builder 与独立审计 `code/artifacts/audit/pi_jwm_p2c_formal_data_audit_20260819_v3b/`：60 条轨迹、36/12/6/6 split、18,000 action attempts、locked-test 未进入 metrics，`formal_data_approved=true`。
- `code/artifacts/formal_tensor/pi_jwm_v4_tensor_v1/` 只 materialize 54 条 train/validation/calibration seed；结构验收和独立 finite/manifest 检查通过，`task_action` 宽度为 8、`task_dag_state` 存在、stats `source_split=train`，未创建任何 locked-test seed tensor。
- 发现并修复两个契约问题：物理状态 phase 混入导致 duplicate snapshot；task snapshot 使用 outcome 时刻导致与 decision grid 不对齐。两者均通过 RED/GREEN 回归后重新采集/审计。
- 新 adapter 的 return action 已使用 `return_target_id`；formal tensor 对已批准的旧 v3 source 显式兼容 `target_node_id`，每个 tensor report 记录兼容计数，不对 source artifact 做静默改写。
- 关键未解决语义阻断：observer 的 `TaskSnapshot` 不含 task_size/return_size/task_cpu/deadline/priority/transmitted/computed/delay 等动态值，adapter 的 task snapshots 因此在 tensor 中产生全零 task_state；observer 生成的 channel_rows 也未被 adapter 映射为 physical-edge feature fields，physical_edge_state 全零。不能从最终 task_records、outcome 或代理量回填，否则会造成未来信息泄漏或理论—实现不一致。
- 结论：`formal_tensor_ready=true` 仅表示结构张量化完成；`formal_training_ready=false` 且训练、GPU、调参和 locked-test 解封继续阻断。下一门是补齐 decision-time direct task/channel field capture，再重新生成非锁定候选并复验。

## 2026-08-19 P2-C 场景冻结前核对

- 当前仓库 HEAD 为 `26d9de7`，工作区无未提交改动；本轮不触碰 `D:\shen\PKU\RRM`。
- `code/src/pi_jwm/formal_airfogsim_dataset_v1.py` 已定义六个候选场景：任务 lambda `0.5/1.0/2.0`，车辆数量上限 `20/40`，车辆到达 lambda `0.5/1.0`；这些值的 `calibration_status` 仍是历史字符串 `calibrated_5s_2seed_20260801`，不能仅据此视为当前正式冻结。
- `code/scripts/calibrate_formal_airfogsim_scenarios_v1.py` 是现有 CPU-only 校准入口；它运行六个场景、每场景多个开发 seed，输出任务数、并发任务、节点/边规模、链路活动率和 CPU 利用率，并只生成 calibration probe，不生成正式数据。
- 既有 `code/artifacts/datasets/airfogsim_formal_v1_calibration/` 报告在 2026-08-01 通过：六场景齐全、每场景 2 个 seed、负载任务数单调、密度节点数单调、观测非空。该报告仍属于候选校准证据，不解除 P2-C 的正式冻结门。
- `config.yaml` 的实际生效参数包括：`traffic.max_n_vehicles`、`traffic.arrival_lambda`、`traffic.UAV_speed_range`、`traffic.max_n_UAVs`、`task.task_generation_kwargs.lambda`、`task_profile.<vehicle|uav>.lambda`、Rayleigh outage 配置和固定 RSU positions。正式配置必须声明哪些是场景因素、哪些保持固定。
- 现有 P2-C candidate JSON 仍将 `scenario_matrix`、`target_scale`、`seed_split` 标为 `not_frozen`；自然 preflight 的 6 条轨迹和 seeds `0/1/2` 不能外推正式规模或正式 split。
- 形式上的正式数据入口仍存在实现门：`code/scripts/build_formal_airfogsim_dataset_v1.py` -> `formal_airfogsim_runtime_v1.py` 当前调用 `task_resource_conservation_audit.run_airfogsim_conservation_seed`，没有接入 P2-B v2 的 ledger-bound full-dual-graph collector，也没有把 `orthogonal/interference_reuse` 资源臂写入正式 `TrajectorySpec`。因此不能在此入口上直接生成或宣称正式 v4。
- P2-B v2 的资源臂由 `full_dual_graph_coverage_v1.choose_resource_arm(trajectory_id, seed)` 按哈希平衡选择；它应作为采集策略/覆盖因素，而不是新增六场景维度。正式协议需要显式记录该臂及其分层平衡规则。

## 2026-08-19 P2-C 协议冻结实现

- 用户确认六场景矩阵、60 条轨迹/30 秒、`36/12/6/6` split、seed 公式和每场景 5/5 资源臂分层。
- `formal_airfogsim_dataset_v1.py` 已冻结 `RESOURCE_ARMS`、保留开发 seed、`10000 + scenario_index*100 + repetition` seed 规则和哈希选择器匹配；每场景资源臂计数、全局计数和跨 split seed 唯一性由 `validate_formal_protocol` 检查。
- `formal_airfogsim_runtime_v1.py` 的输出已携带 `resource_arm`、`balanced_two_arm_v1` 和 collector contract 元数据；当前真实路径明确为 `AirFogSim-Legacy-Conservation-Runner`、`formal_collector_ready=false`。
- `build_formal_airfogsim_dataset_v1.py` 新增安全门：只有 runtime 明确声明 `PIJWM-AirFogSim-Full-Collector-v2` 且 `formal_collector_ready=true`，才允许轨迹通过 `formal_collector_ready` 检查。legacy runner 无法把数据升级为 `formal_dataset_ready`。
- 协议与安全门定向测试已通过；正式 v2 collector 接入和 attempt/reject ledger 持久化仍未完成，因此没有生成正式 v4 数据。
