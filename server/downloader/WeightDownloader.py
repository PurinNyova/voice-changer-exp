import os
from concurrent.futures import ThreadPoolExecutor

from downloader.Downloader import download
from mods.log_control import VoiceChangaerLogger
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams
from Exceptions import WeightDownladException

logger = VoiceChangaerLogger.get_instance().getLogger()


def downloadWeight(voiceChangerParams: VoiceChangerParams):
    content_vec_500_onnx = voiceChangerParams.content_vec_500_onnx
    hubert_base = voiceChangerParams.hubert_base
    light_hubert = voiceChangerParams.light_hubert
    hubert_base_jp = voiceChangerParams.hubert_base_jp
    hubert_soft = voiceChangerParams.hubert_soft
    nsf_hifigan = voiceChangerParams.nsf_hifigan
    crepe_onnx_full = voiceChangerParams.crepe_onnx_full
    crepe_onnx_tiny = voiceChangerParams.crepe_onnx_tiny
    rmvpe = voiceChangerParams.rmvpe
    rmvpe_onnx = voiceChangerParams.rmvpe_onnx
    whisper_tiny = voiceChangerParams.whisper_tiny
    use_light_hubert = bool(light_hubert)

    weight_files = [
        content_vec_500_onnx,
        hubert_base,
        hubert_base_jp,
        hubert_soft,
        nsf_hifigan,
        crepe_onnx_full,
        crepe_onnx_tiny,
        rmvpe,
        whisper_tiny,
    ]
    if use_light_hubert:
        weight_files.append(light_hubert)

    # file exists check (currently only for rvc)
    downloadParams = []
    if os.path.exists(hubert_base) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt",
                "saveTo": hubert_base,
                "position": 0,
            }
        )
    if use_light_hubert and os.path.exists(light_hubert) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/mechanicalsea/lighthubert/resolve/main/lighthubert_base.pt",
                "saveTo": light_hubert,
                "position": 1,
            }
        )
    if os.path.exists(hubert_base_jp) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/rinna/japanese-hubert-base/resolve/main/fairseq/model.pt",
                "saveTo": hubert_base_jp,
                "position": 2,
            }
        )
    if os.path.exists(hubert_soft) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/embedder/hubert-soft-0d54a1f4.pt",
                "saveTo": hubert_soft,
                "position": 3,
            }
        )
    if os.path.exists(nsf_hifigan) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_20221211/model.bin",
                "saveTo": nsf_hifigan,
                "position": 4,
            }
        )
    nsf_hifigan_config = os.path.join(os.path.dirname(nsf_hifigan), "config.json")

    if os.path.exists(nsf_hifigan_config) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/raw/main/ddsp-svc30/nsf_hifigan_20221211/config.json",
                "saveTo": nsf_hifigan_config,
                "position": 5,
            }
        )
    nsf_hifigan_onnx = os.path.join(os.path.dirname(nsf_hifigan), "nsf_hifigan.onnx")
    if os.path.exists(nsf_hifigan_onnx) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/ddsp-svc30/nsf_hifigan_onnx_20221211/nsf_hifigan.onnx",
                "saveTo": nsf_hifigan_onnx,
                "position": 5,
            }
        )

    if os.path.exists(crepe_onnx_full) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/full.onnx",
                "saveTo": crepe_onnx_full,
                "position": 6,
            }
        )
    if os.path.exists(crepe_onnx_tiny) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/crepe/onnx/tiny.onnx",
                "saveTo": crepe_onnx_tiny,
                "position": 7,
            }
        )

    if os.path.exists(content_vec_500_onnx) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx",
                "saveTo": content_vec_500_onnx,
                "position": 8,
            }
        )
    if os.path.exists(rmvpe) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights/resolve/main/rmvpe/rmvpe_20231006.pt",
                "saveTo": rmvpe,
                "position": 9,
            }
        )
    if os.path.exists(rmvpe_onnx) is False:
        downloadParams.append(
            {
                "url": "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx",
                "saveTo": rmvpe_onnx,
                "position": 10,
            }
        )

    if os.path.exists(whisper_tiny) is False:
        downloadParams.append(
            {
                "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
                "saveTo": whisper_tiny,
                "position": 11,
            }
        )

    with ThreadPoolExecutor() as pool:
        pool.map(download, downloadParams)

    if os.path.exists(hubert_base) is False or (use_light_hubert and os.path.exists(light_hubert) is False) or os.path.exists(hubert_base_jp) is False or os.path.exists(hubert_soft) is False or os.path.exists(nsf_hifigan) is False or os.path.exists(nsf_hifigan_config) is False:
        raise WeightDownladException()

    # ファイルサイズをログに書き込む。（デバッグ用）
    for weight in weight_files:
        if os.path.exists(weight):
            file_size = os.path.getsize(weight)
            logger.debug(f"weight file [{weight}]: {file_size}")
        else:
            logger.warning(f"weight file is missing. {weight}")
            raise WeightDownladException()
