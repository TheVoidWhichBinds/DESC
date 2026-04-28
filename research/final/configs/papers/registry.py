from configs.papers.dudt2022_qs import PAPER_CONFIG as DUDT2022_QS










#============== PAPER REGISTRY ===================================================================================
PAPER_REGISTRY = {
    "dudt2022_qs": DUDT2022_QS,
}
#==============================================================================================================










#========== get_paper_config =====================================================================================
def get_paper_config(
        paper_id,
    ):
    if paper_id not in PAPER_REGISTRY:
        available = ", ".join(sorted(PAPER_REGISTRY.keys()))
        raise ValueError(f"Unknown paper_id '{paper_id}'. Available paper IDs: {available}")

    return PAPER_REGISTRY[paper_id]
#==============================================================================================================