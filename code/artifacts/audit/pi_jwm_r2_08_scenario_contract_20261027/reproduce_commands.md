# R2-08 复现命令与边界

工作目录：D:/shen/PKU/PIJWM。Python：D:/miniconda/python.exe。无需安装依赖。禁止启动GPU、训练、原始数据或非train读取。

## 已实际运行

    $env:PYTHONUTF8='1'
    $env:PYTHONDONTWRITEBYTECODE='1'
    python code/scripts/audit_pi_jwm_scenario_contract_r2_08.py
    python code/scripts/audit_pi_jwm_scenario_contract_r2_08.py --output code/artifacts/audit/pi_jwm_r2_08_scenario_contract_20261027/final_acceptance
    python code/scripts/audit_pi_jwm_scenario_contract_r2_08.py --output code/artifacts/audit/pi_jwm_r2_08_scenario_contract_20261027/final_acceptance_v2
    python code/scripts/build_project_knowledge_index_v1.py --check

依次33、35、35项测试通过。根目录首次回执保留；final_acceptance第一次补强回执保留；最终以final_acceptance_v2/result.json及同目录源码指纹为准。再次复现必须使用新的、未含result.json的audit输出目录，不能覆盖上述证据。

只运行test_scenario_contract_v1、test_airfogsim_service_interface_r2_07及test_v2u4real_r2_05b_interface_v1.AirFogSimIngressTests。不运行旧测试类中的真实帧读取。编译采用compile(source_text,path,'exec')，不落盘pycache。

配置文件中的每个非null夹具参数均被标记为ENGINEERING_FIXTURE_ONLY；formal_pending只能结构检查，完整执行按预期被拒绝。保留记录、手工完整输入回执、常数速率和时钟推进都不是任务4生命周期、真实交付或物理证据。

## 一致性与保全

context_consistency_check.json记录9份AI_CONTEXT、20份新增文档前缀、注册表结构与源SHA检查。documentation_patch_manifest.json保存每份前缀和改动前SHA；剥离本轮前缀后逐项比对baseline.json的40项，缺失文件必须仍缺失。注册表历史条目与基线JSON逐项一致，文件格式变化限制为新增条目。

final_acceptance_v2/protection_check.json保存13份R2-07 vendor指纹和73份既有回执前后SHA。原始数据完全不读取，不用文件可读性冒充物理同步。

knowledge_index_check_initial.log和knowledge_index_check_final.log保留只读--check的真实输出。已有五项生成文件不同步时不重建、不push、不绕过hook。提交只含指定R2-08文件；共享脏文档暂存前缀+基线HEAD，保留原有工作区内容。
