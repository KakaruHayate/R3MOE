# **👄 Mouth Opening Baker \- 动画与引擎导入指南**

本工具可以从语音中提取高精度的张嘴幅度参数，并支持导出为多种格式。导出的数据本质上是**基于时间序列的浮点数（范围 0.0 \~ 1.0）**，其中 0.0 代表完全闭嘴，1.0 代表嘴巴张到最大。

以下是针对不同主流软件的接入说明与配套代码：

# **1\. 🎬 Adobe After Effects (AE)**

**适用场景**：MG动画、二维扁平角色、通过时间重映射（Time Remapping）驱动的序列帧口型。

**使用文件**：xxx.json

## **接入步骤：**

1. 将工具导出的 .json 文件拖入 AE 的**项目面板（Project）**。  
2. 将该 .json 文件拖入你的\*\*合成序列（Timeline）\*\*中，确保它和你的音频对齐。  
3. 找到你想要驱动的图层属性（例如：嘴巴图层的**缩放 Scale**，或者预合成的**时间重映射 Time Remap**）。  
4. 按住键盘上的 Alt 键（Mac为 Option 键），鼠标左键点击该属性旁边的**秒表图标**，开启表达式输入框。  
5. 将以下代码复制并粘贴进去，根据注释修改参数即可。

## **AE 表达式代码 (以驱动 Y 轴缩放为例)：**
```
// 1. 绑定你的 JSON 数据层（请将名字修改为你的 json 文件名）
var jsonLayer = thisComp.layer("your_audio_mouth.json"); 
var fps = jsonLayer.sourceData.fps;
var data = jsonLayer.sourceData.data;

// 2. 将当前合成时间换算为数据帧数
var currentFrame = Math.floor(time * fps);

// 3. 安全钳制：防止数组越界导致表达式报错
currentFrame = Math.max(0, Math.min(currentFrame, data.length - 1));

// 4. 获取当前时间的口型数值 (0.0 ~ 1.0)
var mouthValue = data[currentFrame];

// 5. 【自定义区域】数值映射 (Mapping)
var closeMouthScaleY = 20;  // 闭嘴时，嘴巴的 Y轴缩放值 (如 20%)
var openMouthScaleY = 100;  // 张到最大时，嘴巴的 Y轴缩放值 (如 100%)

// 计算最终的 Y 轴缩放
var finalY = closeMouthScaleY + (openMouthScaleY - closeMouthScaleY) * mouthValue;

// 输出结果：[X轴保持原始设定, Y轴跟随声音跳动]
[value[0], finalY];
```
# **2\. 🎮 Unity 3D / Live2D SDK**

**适用场景**：3D 游戏角色开发、VTube 虚拟主播、实时互动应用。

**使用文件**：xxx.csv

## **接入步骤：**

1. 将导出的 .csv 文件和音频文件一并导入 Unity 项目的 Assets 目录。  
2. 在模型（或 Live2D 预制体）上新建一个 C\# 脚本，命名为 MouthController.cs。  
3. 将以下代码复制进脚本。  
4. 在 Unity 的 Inspector 面板中，把 CSV 文件拖给 Csv File，绑定你的 Audio Source，并指定对应的 Skinned Mesh Renderer（含有 BlendShape 的模型）。

## **Unity C\# 驱动脚本：**
```
using UnityEngine;
using System.Globalization;

public class MouthController : MonoBehaviour
{
    [Header("Data & Audio")]
    public TextAsset csvFile;          // 拖入导出的 CSV 文件
    public AudioSource audioSource;    // 对应的音频源
    public float targetFPS = 30f;      // 提取时设置的 FPS，默认 30

    [Header("3D Model Setup")]
    public SkinnedMeshRenderer smr;    // 含有嘴部表情的模型
    public int blendShapeIndex = 0;    // 张嘴动作在 BlendShape 列表里的序号
    public float maxBlendShapeWeight = 100f; // 最大权重（Unity默认为100）

    private float[] mouthData;

    void Start()
    {
        ParseCSV();
    }

    void ParseCSV()
    {
        if (csvFile == null) return;

        // 按行分割 CSV
        string[] lines = csvFile.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        
        // 第一行是表头 ("MouthOpening")，所以数据长度是 lines.Length - 1
        mouthData = new float[lines.Length - 1]; 
        
        for (int i = 1; i < lines.Length; i++)
        {
            if (float.TryParse(lines[i], NumberStyles.Any, CultureInfo.InvariantCulture, out float val))
            {
                mouthData[i - 1] = val;
            }
        }
    }

    void Update()
    {
        // 只有当声音在播放，且有数据时才驱动
        if (audioSource != null && audioSource.isPlaying && mouthData != null && mouthData.Length > 0)
        {
            // 通过音频精确播放时间来锁定数据帧，绝对不发生音画错位
            int currentFrame = Mathf.FloorToInt(audioSource.time * targetFPS);
            currentFrame = Mathf.Clamp(currentFrame, 0, mouthData.Length - 1);
            
            // 计算当前权重：0.0~1.0 映射到 0~100
            float weight = mouthData[currentFrame] * maxBlendShapeWeight;
            
            // 驱动 3D 模型
            if (smr != null)
            {
                smr.SetBlendShapeWeight(blendShapeIndex, weight);
            }
            
            /* * 如果你使用的是 Live2D Cubism SDK，请注释掉上面的 SMR 代码，使用下方代码：
             * GetComponent<CubismModel>().Parameters[0].Value = mouthData[currentFrame];
             */
        }
    }
}
```
# **3\. 🐒 Blender (3D 动画制作)**

**适用场景**：3D 动画短片、MMD渲染、离线关键帧动画修整。

**使用文件**：xxx.csv

在 Blender 中，由于逐帧读取文件效率较低，动画师通常喜欢将数据直接\*\*烘焙（Bake）\*\*成时间轴上的关键帧（Keyframes），这样方便后续手动 K 帧微调。

## **接入步骤：**

1. 打开 Blender，进入 Scripting（脚本编辑）工作区。  
2. 新建一个文本文件，将下方代码复制进去。  
3. 修改代码顶部【需要修改的参数】区域。  
4. 点击顶部的 **运行脚本 (Run Script)** 按钮（或按 Alt \+ P）。  
5. 此时时间轴上会自动打满对应形态键（Shape Key）的关键帧。

## **Blender Python 烘焙脚本：**
```
import bpy

# ================= 需要修改的参数 =================
# 1. 导出的 CSV 文件绝对路径 (注意使用正斜杠 / )
csv_path = "C:/Users/YourName/Desktop/output_mouth.csv"

# 2. 你的 3D 模型名字
obj_name = "FaceMesh"

# 3. 控制张嘴的形态键 (Shape Key) 名字
shape_key_name = "MouthOpen"

# 4. 提取数据时的 FPS
data_fps = 30

# 5. 起始帧 (如果你希望动画从第 100 帧开始播放，填 100)
start_frame = 1
# ==================================================

def bake_mouth_data():
    obj = bpy.data.objects.get(obj_name)
    if not obj or not obj.data.shape_keys:
        print(f"错误：找不到模型 '{obj_name}' 或其没有形态键！")
        return
        
    shape_keys = obj.data.shape_keys.key_blocks
    if shape_key_name not in shape_keys:
        print(f"错误：找不到名为 '{shape_key_name}' 的形态键！")
        return
        
    mouth_key = shape_keys[shape_key_name]
    
    # 强制匹配场景帧率以保证音画同步
    bpy.context.scene.render.fps = data_fps

    # 读取 CSV 并烘焙关键帧
    try:
        with open(csv_path, 'r') as file:
            lines = file.readlines()[1:] # 跳过表头 "MouthOpening"
            
            print(f"开始烘焙 {len(lines)} 帧数据...")
            for i, line in enumerate(lines):
                val_str = line.strip()
                if not val_str: continue
                
                # 读取 0~1 的数值
                val = float(val_str)
                
                # 赋值并插入关键帧
                current_frame = start_frame + i
                mouth_key.value = val
                mouth_key.keyframe_insert(data_path='value', frame=current_frame)
                
        print("✅ 口型关键帧烘焙完成！请在 Timeline 查看。")
        
    except FileNotFoundError:
        print(f"错误：找不到 CSV 文件，请检查路径: {csv_path}")

# 执行函数
bake_mouth_data()
```
