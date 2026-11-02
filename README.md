![Comp 3_04707](https://github.com/user-attachments/assets/f5a7732f-d721-4ab4-905e-7fe7646abe9a)

[![Version](https://img.shields.io/badge/version-1.0-2ea44f?style=for-the-badge)](https://github.com/MashiroW/openGL-WorldEditor) [![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0wIDBoMTF2MTFIMHpNMTMgMGgxMXYxMUgxM3pNMCAxM2gxMXYxMUgwek0xMyAxM2gxMXYxMUgxM3oiLz48L3N2Zz4=&logoColor=white)](#) [![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)](#) [![macOS](https://img.shields.io/badge/macOS-000000?style=for-the-badge&logo=apple&logoColor=white)](#) [![Stars](https://img.shields.io/github/stars/MashiroW/openGL-WorldEditor?style=for-the-badge&logo=github&color=e3b341)](https://github.com/MashiroW/openGL-WorldEditor/stargazers) [![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white)](https://github.com/sponsors/MashiroW)

# 📦 Requirements

This project requires **Python 3.8+** and the following packages
You can install all dependencies using:

```bash
pip install -r requirements.txt
```

# ✨ Features

🗺️ Scene Management  
- Load and save scenes to/from disk  
- Add or remove models dynamically  
- Translate and rotate objects in the scene  
- Visualize a built-in starry skybox (not editable yet)  
- Translate objects using on-screen gizmos (mouse-controlled)  

🧱 Model Support  
- Load models in .obj format  
- Parse and apply vertex colors from .mtl files  
- Load and render textures in .png format  

💡 Lighting System  
- Add custom light sources to your scene  
- Control light color, radius, and brightness  
- Real-time lighting updates in the viewport  

🎥 Camera Modes  
- FPS-style camera movement for free exploration  
- Optional camera auto-rotation (1-axis)  

🔗 Library Integration  
- The core features of the project can be easily reused as a standalone **OpenGL world editing library**.  
- An example of such integration can be found [here](https://github.com/MashiroW/wc3-language-patcher), where a scene was designed using this editor, then saved to disk and later reloaded using the **OpenGL rendering output and scene-loading functions** provided by this project, to render the scene independently in the linked program.

# Videos

## Free camera
  
https://github.com/user-attachments/assets/fbc6a806-e263-4d54-8812-c03a462a4949

## Light sources, Gizmo transforms
  
https://github.com/user-attachments/assets/bbdb655f-7e8e-48ea-a5a5-8c19a1461d70
