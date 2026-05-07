from PIL import Image
import torch
import re
import base64
from io import BytesIO
import os

def pil_to_data_url(pil_image, default_format="jpeg"):
    """PIL Image对象转DataURL（格式：data:image/xxx;base64,xxx）"""
    try:
        # 1. 确定图片格式（优先用PIL图片自带格式，无则用默认格式）
        img_format = pil_image.format.lower() if pil_image.format else default_format
        img_format = img_format.lower()  # 统一转为小写（如JPEG→jpeg）

        # 2. 处理格式兼容问题（JPEG不支持透明通道）
        if img_format == "jpeg" and pil_image.mode in ("RGBA", "P", "LA"):
            pil_image = pil_image.convert("RGB")  # 转为RGB模式

        # 3. 将PIL图片写入字节流
        buffer = BytesIO()
        # 保存时指定格式（需大写），PNG保留透明通道
        save_params = {"format": img_format.upper()}
        if img_format == "png" and pil_image.mode == "RGBA":
            save_params["transparency"] = 0  # 保留透明
        pil_image.save(buffer, **save_params)

        # 4. 字节流编码为Base64字符串
        buffer.seek(0)  # 重置指针到开头
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # 5. 拼接成DataURL
        return f"data:image/{img_format};base64,{base64_str}"
    except Exception as e:
        print(f"PIL转DataURL失败：{str(e)}")
        return None

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

class SeedScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
    @torch.no_grad()
    def __call__(self, prompt, images):
        contents = [{"type": "image_url", "image_url": pil_to_data_url(image)} for image in images]
        reward_prompt = f"""
        为以上8张图计算（Image Reward）评分，原始提示词为{prompt}

        从美学表现力，画面合理性，提示词相关性，人物细节，文字书写准确性和画面创意度和真实场景还原度七个核心维度进行评分，总分35分。
        这套标准参考了目前主流的 AI 绘画评价模型（如 PickScore, ImageReward, HPS v2），
        旨在为模型微调或人工标注提供清晰的指引。
        
        1. 美学表现力 (Aesthetic Quality)关注点： 光影、构图、色彩、艺术感染力及整体视觉冲击力。
        1分（差）： 极度模糊，曝光不足或严重过曝，噪点满布，主体完全无法辨认，构图毫无逻辑，视觉观感极差。
        2分（较差）： 画质低劣，色彩极度不协调，光影生硬，构图支离破碎，缺乏任何审美价值。
        3分（一般）： 画质合格，对焦尚可，色彩与光效处于平均水平，构图普通（如简单的居中），缺乏艺术感和吸引力。
        4分（好）： 图像清晰锐利，光影层次分明，色彩搭配悦目，构图符合摄影或绘画美学规律，具有一定的视觉美感。
        5分（优）： 极致的清晰度与细节展现，光影运用大师级，色彩富有情感表达，构图极具创意或视觉张力，具备艺术品般的感染力。
        2. 画面合理性 (Image Rationality/Coherence)关注点： 逻辑一致性、物体结构、物理规律、是否存在AI伪影（Artifacts）。
        1分（差）： 逻辑完全崩坏（如物体漂浮、重叠、断裂），充满严重的AI伪影，完全无法理解画面表达的内容。
        2分（较差）： 存在明显的逻辑错误（如三条腿的桌子、融化的背景），物体比例严重失调，物理关系违背常识。
        3分（一般）： 基本逻辑通顺，但细看存在小瑕疵（如物体交界处模糊、透视略显别扭），画面略显僵硬但不影响整体理解。
        4分（好）： 逻辑清晰，物体结构完整且比例协调，透视关系正确，没有明显的伪影或违和感。
        5分（优）： 完美的逻辑一致性，细节处经得起推敲（如影子方向、折射反射等物理现象），画面自然且极具真实感/完整感。
        3. 提示词相关性 (Prompt Alignment)关注点： 图像内容与提示词的语义契合度。评估模型是否遗漏了主体、场景、动作、风格或数量。
        1分（差）： 完全离题。 画面内容与提示词描述完全不符，没有体现出任何关键词。
        2分（较差）： 捕捉碎片。 仅体现了次要关键词（如背景颜色），但遗漏了核心主体，或完全误解了提示词的风格要求。
        3分（一般）： 核心符合。 包含了核心主体和场景，但遗漏了重要的修饰语（如动作、光影要求、特定的物体数量）。
        4分（好）： 高度契合。 准确还原了提示词中的所有主要元素和细节，场景、动作和环境描述清晰可见。
        5分（优）： 极致还原。 不仅捕捉到了所有显性特征，还精准呈现了提示词隐含的情绪、氛围及复杂的逻辑关系（如：A在B的左边并看着C）。
        4. 人物细节 (Human Detail/Anatomy)关注点： 肢体结构（尤其是手指）、面部五官、皮肤质感、眼神及其自然度。
        1分（差）： 人物完全畸形，五官错位，肢体严重扭曲（如多手多脚），“克苏鲁”风格的崩坏感。
        2分（较差）： 存在明显的人体结构错误（如手指融合、关节反折），五官僵硬或变形，表情极不自然。
        3分（一般）： 人物形态基本完整，但细节粗糙。手部可能存在轻微不自然，眼神略显空洞，皮肤纹理缺乏细节。
        4分（好）： 肢体结构正确，手部细节基本正常，五官精致且对称，表情生动，皮肤质感较为真实。
        5分（优）： 完美的人体解剖结构，手部细节（指甲、纹路）完美无瑕，眼神深邃传神，皮肤毛孔、发丝等细节清晰可见，极具生命力。
        5. 书写准确性 (Text Writing Accuracy)关注点： 画面中文字字符的渲染质量。评估是否存在拼写错误、乱码、笔画畸形或语言类型错误。
        1分（差）： 不可辨认。 要求的文字完全没出现，或呈现为毫无规律的乱码、线条堆叠或“外星文字”。
        2分（较差）： 严重畸形。 文字勉强有字形，但存在严重拼写错误（如多/漏字母），笔画严重断裂、重叠或左右颠倒。
        3分（一般）： 基本可读。 核心词汇可以辨认，但存在细微笔画瑕疵（如字符粘连、o和e不分）或轻微拼写错误（如 "Apple" 写成 "Aple"）。
        4分（好）： 拼写正确。 文字拼写完全正确，字迹清晰，字体风格自然，无明显渲染伪影。
        5分（优）： 完美排版。 拼写无误且美观，文字与画面环境（如材质、阴影、透视、光反射）完美融合，宛如实拍或专业设计。
        6. 画面创意度 (Image Creativity / Originality)关注点： 构思的新颖性、视觉隐喻的应用、艺术风格的独特性，以及是否具备如电影般的视觉吸引力和“破圈”潜力。
        1分（差）： 视觉平庸。 毫无构思，像是一张随意的生活快照或最基础的素材堆砌，画面元素散乱，完全没有设计感。
        2分（较差）： 陈词滥调。 构图极其陈旧（如简单的居中对齐），色彩和风格大众化，缺乏视觉抓手（Hook），无法引起观众的兴趣。
        3分（一般）： 中规中矩。 画面完整且符合基本美学，但缺乏惊喜感。类似于常见的、合格的商业素材图，完成了任务但没有亮点。
        4分（好）： 构思巧妙。 具有明显的视觉主题，能运用有趣的构图、光影对比或独特的艺术风格来表达内容，具备优秀海报的潜质。
        5分（优）： 惊艳之作。 构思极其精妙且富有想象力，具备强烈的视觉冲击力和深刻的视觉隐喻。艺术表现力突破常规，达到顶尖商业海报或艺术作品的水平，令人过目不忘。
        7. 真实场景还原度 (Realism / Authenticity) 在 AI 绘画领域，我们常说的“AI 感”通常指：过分光滑的塑料质感、超现实的过饱和色彩、过于完美的对称脸，以及那种典型的“Midjourney 式”油画光影。
        增加这个维度能有效过滤掉“一眼假”的生成图，对于追求写实摄影、监控视角或机器人视觉模拟等场景尤其重要。
        关注点： 画面是否具备真实摄影的质感，是否成功规避了“塑料感”和“恐怖谷效应”。评估皮肤纹理、环境杂质、镜头瑕疵（如噪点、色散）及自然光影的真实性。
        1分（差）： 强烈的“AI 塑料感”。 画面呈现出廉价的 3D 建模感或过度磨皮的质感。物体边缘有奇怪的荧光，人物皮肤像蜡像，材质完全不符合物理光学，存在明显的“恐怖谷”特征。
        2分（较差）： 明显的合成痕迹。 色彩过分饱和且生硬，光影方向混乱，画面过于“干净”而显得虚假。
        3分（一般）： 半写实状态。 猛一看像照片，但细节经不起推敲。纹理过于均匀重复，缺乏现实世界应有的微小瑕疵（如灰尘、划痕、皮肤毛孔），光影略显死板。
        4分（好）： 高度写实。 具备真实的摄影特征（如自然的景深、镜头暗角或微小噪点）。材质纹理生动（如金属的锈迹、布料的纤维），光影关系符合自然规律。
        5分（优）： 难辨真伪（Indistinguishable）。 完美复刻了真实相机的成像质量。包含了丰富的随机性细节（如细微的乱发、皮肤的不均色斑、复杂的反射光），完全没有 AI 生成的痕迹，视觉观感等同于实拍照片。
        
        人物细节，文字书写准确性在提示词中没有文字或者人物要求的情况下取均分3.5。
        海报设计类提示词，画面创意度的区分度要高，其他如忠实还原具体人物或场景的摄影类提示词，画面创意度不做要求，统一3.5。
        摄影类提示词，真实场景还原度区分度要高，这个与画面创意度可能存在矛盾，其他类提示词（如动漫作品）该项统一3.5）。
        
        list返回最终分数(8个英文逗号隔开纯数字输出，不要额外东西.也不要外层[])
        """
        contents.append({"type": "text", "text": reward_prompt})
        from openai import OpenAI
        client = OpenAI(
            api_key = os.getenv("SEED_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        MAX_TRY = 5
        while MAX_TRY:
            try:
                completion = client.chat.completions.create(
                    model="doubao-seed-1-8-251228",
                    messages=[{"role": "user", "content": contents}],
                    max_completion_tokens=65535,
                    reasoning_effort="medium"
                )

                rewards = completion.choices[0].message.content
                final_rewards = [float(r) for r in rewards.split(',')]
                assert len(final_rewards) == 8
                break
            except:
                MAX_TRY -= 1
                print('retry')
        return final_rewards

# Usage example
def main():
    scorer = SeedScorer(
        device="cuda",
        dtype=torch.bfloat16
    )
    image_dir = '/workspace/flow_grpo/wandb/run-20260221_194854-hyf2x29l/files/media/images'
    images= [os.path.join(image_dir,image_name) for image_name in os.listdir(image_dir)]
    pil_images = [Image.open(img) for img in images]

    prompt = 'The very simple shape of the smiling face of the original Logo is retained, but the characteristic l'
    print(scorer(prompt, pil_images))

if __name__ == "__main__":
    main()