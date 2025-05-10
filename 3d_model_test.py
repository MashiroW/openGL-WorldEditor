import sys
import numpy as np
import os
import random
import math
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                           QWidget, QFileDialog, QHBoxLayout, QLabel, QDoubleSpinBox,
                           QSlider, QCheckBox, QGroupBox, QFormLayout, QColorDialog,
                           QTabWidget, QComboBox)
#from OpenGL.constants import GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3, GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QColor
from PyQt5.QtOpenGL import QGLWidget
from OpenGL.GL import *
from OpenGL.GLU import *
from collections import OrderedDict
from OpenGL.GL import (
    GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3, 
    GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7,
    glIsEnabled
)
from OpenGL.GL import GL_UNSIGNED_INT, GL_FRONT_AND_BACK
from OpenGL.GL import (
    GL_TEXTURE_2D, GL_RGB, GL_UNSIGNED_BYTE, GL_TEXTURE_MIN_FILTER, 
    GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T, 
    GL_REPEAT, glGenTextures, glBindTexture, glTexImage2D, glTexParameteri,
    glEnable, glDisable
)

from PyQt5.QtWidgets import QListWidget
from PyQt5.QtGui import QSurfaceFormat
from PIL import Image

from OpenGL.GL import GL_LINES, GL_LINE_WIDTH

# Fix for high DPI displays
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

class StarBackground:
    def __init__(self, count=2000):  # Increased star count
        self.stars = []
        self.generate_stars(count)
        
    def generate_stars(self, count):
        self.stars = []
        for _ in range(count):
            theta = random.uniform(0, 2 * math.pi)
            phi = math.acos(1 - 2 * random.uniform(0, 1))
            r = 100.0
            
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.sin(phi) * math.sin(theta)
            z = r * math.cos(phi)
            
            brightness = random.uniform(0.3, 1.0)
            self.stars.append((x, y, z, brightness))  # Only 4 values now
    
    def draw(self, camera_pos, camera_pitch, camera_yaw, model_angle):
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        # Apply camera rotation to the stars (but not position)
        glRotatef(camera_pitch * 0.5, 1, 0, 0)  # Reduced vertical movement
        glRotatef(camera_yaw * 0.5, 0, 1, 0)    # Reduced horizontal movement
        glRotatef(model_angle * 0.1, 0, 1, 0)   # Slow background rotation
        
        glPointSize(2.0)
        glBegin(GL_POINTS)
        for x, y, z, brightness in self.stars:
            glColor4f(1.0, 1.0, 1.0, brightness)
            glVertex3f(x, y, z)
        glEnd()
        
        glPopMatrix()
        glPopAttrib()

class ModelObject:
    def __init__(self, filename=""):
        self.render_objects = []
        self.position = [0.0, 0.0, 0.0]
        self.rotation = [0.0, 0.0, 0.0]
        self.scale = 1.0
        self.visible = True
        self.wireframe = False
        self.filename = filename
        self.texture = None
        self.internal_name = ""  # Will be used for JSON key
        self.display_name = ""   # Will be shown in GUI
        
        if filename:
            self.load_obj(filename)
            self.try_load_texture()

    def calculate_bounding_box(self):
        if not self.render_objects:
            return None
        
        # Initialize min/max with first vertex
        min_coords = None
        max_coords = None
        
        # Find min/max across all vertices
        for obj in self.render_objects:
            for vertex in obj['vertices']:
                if min_coords is None:
                    min_coords = np.array(vertex)
                    max_coords = np.array(vertex)
                else:
                    min_coords = np.minimum(min_coords, vertex)
                    max_coords = np.maximum(max_coords, vertex)
        
        return min_coords, max_coords

    def center_model(self):
        min_coords, max_coords = self.calculate_bounding_box()
        if min_coords is None or max_coords is None:
            return
        
        # Calculate center
        center = (min_coords + max_coords) / 2.0
        
        # Offset all vertices
        for obj in self.render_objects:
            # Convert vertices to numpy array if they aren't already
            if not isinstance(obj['vertices'], np.ndarray):
                obj['vertices'] = np.array(obj['vertices'], dtype=np.float32)
            
            # Subtract center from all vertices
            obj['vertices'] -= center
        
        # Update position to maintain world position
        self.position = center.tolist()
        
        # Also center the texture coordinates if they exist
        for obj in self.render_objects:
            if obj['tex_coords'] is not None:
                # Convert to numpy array if needed
                if not isinstance(obj['tex_coords'], np.ndarray):
                    obj['tex_coords'] = np.array(obj['tex_coords'], dtype=np.float32)
                
                # Calculate texture coordinate center
                tex_min = np.min(obj['tex_coords'], axis=0)
                tex_max = np.max(obj['tex_coords'], axis=0)
                tex_center = (tex_min + tex_max) / 2.0
                
                # Center texture coordinates
                obj['tex_coords'] -= tex_center
                obj['tex_coords'] += 0.5  # Move to typical texture coordinate range [0,1]

    def load_obj(self, filename):
        try:
            vertices = []
            tex_coords = []
            normals = []
            faces = []
            materials = {}
            current_material = None
            
            # Try to load MTL file if it exists
            mtl_filename = os.path.splitext(filename)[0] + '.mtl'
            if os.path.exists(mtl_filename):
                with open(mtl_filename, 'r') as mtl_file:
                    current_mtl = None
                    for line in mtl_file:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        
                        if parts[0] == 'newmtl':
                            current_mtl = parts[1]
                            materials[current_mtl] = {
                                'ambient': [0.2, 0.2, 0.2],
                                'diffuse': [0.8, 0.8, 0.8],
                                'specular': [0.0, 0.0, 0.0],
                                'shininess': 0.0
                            }
                        elif current_mtl:
                            if parts[0] == 'Ka':
                                materials[current_mtl]['ambient'] = list(map(float, parts[1:4]))
                            elif parts[0] == 'Kd':
                                materials[current_mtl]['diffuse'] = list(map(float, parts[1:4]))
                            elif parts[0] == 'Ks':
                                materials[current_mtl]['specular'] = list(map(float, parts[1:4]))
                            elif parts[0] == 'Ns':
                                materials[current_mtl]['shininess'] = float(parts[1])
            
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    
                    if parts[0] == 'v':
                        vertices.append(list(map(float, parts[1:4])))
                    elif parts[0] == 'vt':
                        tex_coords.append(list(map(float, parts[1:3])))
                    elif parts[0] == 'vn':
                        normals.append(list(map(float, parts[1:4])))
                    elif parts[0] == 'usemtl':
                        current_material = parts[1]
                    elif parts[0] == 'f':
                        face_vertices = []
                        face_tex_coords = []
                        face_normals = []
                        
                        for part in parts[1:]:
                            indices = part.split('/')
                            v_idx = int(indices[0]) - 1 if indices[0] else -1
                            t_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
                            n_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else -1
                            
                            face_vertices.append(v_idx)
                            if t_idx != -1:
                                face_tex_coords.append(t_idx)
                            if n_idx != -1:
                                face_normals.append(n_idx)
                        
                        # Only process faces with 3 or more vertices
                        if len(face_vertices) >= 3:
                            # Triangulate the face (works for both triangles and quads)
                            for i in range(1, len(face_vertices) - 1):
                                faces.append({
                                    'vertex_indices': [face_vertices[0], face_vertices[i], face_vertices[i+1]],
                                    'tex_coord_indices': (
                                        [face_tex_coords[0], face_tex_coords[i], face_tex_coords[i+1]] 
                                        if face_tex_coords else None
                                    ),
                                    'normal_indices': (
                                        [face_normals[0], face_normals[i], face_normals[i+1]] 
                                        if face_normals else None
                                    ),
                                    'material': current_material
                                })
            
            vertices = np.array(vertices, dtype=np.float32)
            tex_coords = np.array(tex_coords, dtype=np.float32) if tex_coords else None
            normals = np.array(normals, dtype=np.float32) if normals else None
            
            # Group faces by material
            material_groups = {}
            for face in faces:
                mat_name = face['material'] or 'default'
                if mat_name not in material_groups:
                    material_groups[mat_name] = []
                material_groups[mat_name].append(face)
            
            # Create render objects for each material group
            self.render_objects = []
            for mat_name, mat_faces in material_groups.items():
                # Get material properties or use defaults
                mat_props = materials.get(mat_name, {
                    'ambient': [0.2, 0.2, 0.2],
                    'diffuse': [0.8, 0.8, 0.8],
                    'specular': [0.0, 0.0, 0.0],
                    'shininess': 0.0
                })
                
                # Flatten face indices
                vertex_indices = np.array([vi for f in mat_faces for vi in f['vertex_indices']], dtype=np.uint32)
                tex_coord_indices = (
                    np.array([ti for f in mat_faces for ti in f['tex_coord_indices']], dtype=np.uint32) 
                    if mat_faces[0]['tex_coord_indices'] else None
                )
                normal_indices = (
                    np.array([ni for f in mat_faces for ni in f['normal_indices']], dtype=np.uint32) 
                    if mat_faces[0]['normal_indices'] else None
                )
                
                self.render_objects.append({
                    'vertices': vertices,
                    'tex_coords': tex_coords,
                    'normals': normals,
                    'vertex_indices': vertex_indices,
                    'tex_coord_indices': tex_coord_indices,
                    'normal_indices': normal_indices,
                    'ambient': mat_props['ambient'],
                    'diffuse': mat_props['diffuse'],
                    'specular': mat_props['specular'],
                    'shininess': mat_props['shininess']
                })
            
            if vertices:
                self.center_model()
            
            return True
        except Exception as e:
            print(f"Error loading OBJ: {e}")
            return False
    
    def calculate_normals(self):
        for obj in self.render_objects:
            vertices = obj['vertices']
            indices = obj['vertex_indices']
            
            # Create zeroed normals array
            normals = np.zeros_like(vertices)
            
            # Calculate face normals and accumulate
            for i in range(0, len(indices), 3):
                v1 = vertices[indices[i]]
                v2 = vertices[indices[i+1]]
                v3 = vertices[indices[i+2]]
                
                normal = np.cross(v2-v1, v3-v1)
                normal /= np.linalg.norm(normal)
                
                normals[indices[i]] += normal
                normals[indices[i+1]] += normal
                normals[indices[i+2]] += normal
            
            # Normalize vertex normals
            norms = np.linalg.norm(normals, axis=1)
            normals = normals / norms[:, np.newaxis]
            
            obj['normals'] = normals.astype(np.float32)

    def try_load_texture(self):
        if not self.filename:
            return False
            
        # Try common texture extensions
        base_path = os.path.splitext(self.filename)[0]
        texture_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tga']
        
        for ext in texture_extensions:
            texture_path = base_path + ext
            if os.path.exists(texture_path):
                try:
                    self.load_texture(texture_path)
                    return True
                except Exception as e:
                    print(f"Failed to load texture {texture_path}: {e}")
                    continue
        return False

    def load_texture(self, texture_path):
        try:
            # Load image file
            image = Image.open(texture_path)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            img_data = image.convert("RGB").tobytes()
            
            # Create OpenGL texture
            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height,
                         0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            
            # Set texture parameters
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            
            self.texture = texture_id
            return True
        except Exception as e:
            print(f"Error loading texture: {e}")
            return False

    def draw(self):
        if not self.visible or not self.render_objects:
            return
            
        glPushMatrix()
        glTranslatef(*self.position)
        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)
        glScalef(self.scale, self.scale, self.scale)
        
        # Enable texturing if we have a texture
        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture)
        
        for obj in self.render_objects:

            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Material properties
            ambient = [c * 0.5 for c in obj['diffuse']]
            ambient = (GLfloat * 4)(*ambient, 1.0)
            
            diffuse = (GLfloat * 4)(*obj['diffuse'], 1.0)
            specular = (GLfloat * 4)(*obj['specular'], 1.0)
            
            glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient)
            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse)
            glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular)
            glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, max(0.0, min(obj['shininess'], 128.0)))
            
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, obj['vertices'])
            
            if obj['normals'] is not None:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, obj['normals'])
            
            if obj['tex_coords'] is not None and self.texture:
                glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                glTexCoordPointer(2, GL_FLOAT, 0, obj['tex_coords'])
            
            glDrawElements(GL_TRIANGLES, len(obj['vertex_indices']), GL_UNSIGNED_INT, obj['vertex_indices'])
            
            # Clean up
            if obj['normals'] is not None:
                glDisableClientState(GL_NORMAL_ARRAY)
            if obj['tex_coords'] is not None and self.texture:
                glDisableClientState(GL_TEXTURE_COORD_ARRAY)
            glDisableClientState(GL_VERTEX_ARRAY)
            
            if self.wireframe:
                glPushAttrib(GL_CURRENT_BIT | GL_LIGHTING_BIT | GL_TEXTURE_BIT)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glDisable(GL_LIGHTING)
                glDisable(GL_TEXTURE_2D)
                glColor3f(0.1, 0.1, 0.1)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, obj['vertices'])
                glDrawElements(GL_TRIANGLES, len(obj['vertex_indices']), GL_UNSIGNED_INT, obj['vertex_indices'])
                glDisableClientState(GL_VERTEX_ARRAY)
                glPopAttrib()
        
        # Disable texturing if it was enabled
        if self.texture:
            glDisable(GL_TEXTURE_2D)
        
        glPopMatrix()

class LightObject:
    def __init__(self):
        self.position = [1.0, 1.0, 1.0]
        self.color = [0.9, 0.8, 0.6]  # Warm yellow default like old version
        self.brightness = 1.0
        self.range = 1.0
        self.enabled = True
        self.show_marker = True
        self.light_id = None
    
    def setup_light(self, light_id):
        self.light_id = light_id
        glEnable(light_id)
        self.update_light()
    
    def update_light(self):
        if not self.light_id:
            return
            
        if self.enabled:
            # Only update light properties if it's enabled
            ambient = [c * 0.1 * self.brightness for c in self.color]
            diffuse = [c * self.brightness for c in self.color]
            specular = diffuse.copy()
            
            glLightfv(self.light_id, GL_POSITION, [*self.position, 1.0])
            glLightfv(self.light_id, GL_AMBIENT, ambient)
            glLightfv(self.light_id, GL_DIFFUSE, diffuse)
            glLightfv(self.light_id, GL_SPECULAR, specular)
            glLightf(self.light_id, GL_CONSTANT_ATTENUATION, 1.0)
            glLightf(self.light_id, GL_LINEAR_ATTENUATION, 0.7 / self.range)
            glLightf(self.light_id, GL_QUADRATIC_ATTENUATION, 1.8 / (self.range * self.range))
    
    def draw_marker(self):
        if not self.show_marker or not self.enabled:
            return
            
        glPushAttrib(GL_ALL_ATTRIB_BITS)
        glDisable(GL_LIGHTING)
        glColor3f(*self.color)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glTranslatef(*self.position)
        
        sphere = gluNewQuadric()
        gluSphere(sphere, 0.1, 10, 10)
        gluDeleteQuadric(sphere)
        
        glBegin(GL_LINES)
        glVertex3f(-0.2, 0, 0)
        glVertex3f(0.2, 0, 0)
        glVertex3f(0, -0.2, 0)
        glVertex3f(0, 0.2, 0)
        glVertex3f(0, 0, -0.2)
        glVertex3f(0, 0, 0.2)
        glEnd()
        
        glPopMatrix()
        glPopAttrib()

class OpenGLWidget(QGLWidget):
    def __init__(self, parent=None):
        fmt = QSurfaceFormat()
        fmt.setSamples(4)  # 4x MSAA
        QSurfaceFormat.setDefaultFormat(fmt)
        
        super(OpenGLWidget, self).__init__(parent)
        
        # Rest of your initialization code...
        self.models = OrderedDict()
        self.lights = OrderedDict()
        self.angle = 0
        self.star_background = None
        
        self.camera_distance = 5.0
        self.camera_rot_x = 30.0
        self.camera_rot_y = 30.0
        self.camera_pos = [0.0, 0.0, 0.0]
        
        self.last_pos = QPoint()
        self.rotating = False
        self.panning = False
        
        self.use_lighting = True
        self.auto_rotate = True
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.auto_rotate_update)
        self.timer.start(16)
        
        self.parent_window = parent

        self.camera_speed = 0.1
        self.camera_sensitivity = 0.5
        self.camera_pitch = 0.0
        self.camera_yaw = -90.0  # Start looking along -Z axis
        self.forward = False
        self.backward = False
        self.left = False
        self.right = False
        self.up = False
        self.down = False

        self.setFocusPolicy(Qt.StrongFocus)

    def initializeGL(self):
        # Enable multisampling
        glEnable(GL_MULTISAMPLE)
        
        # Other quality settings
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glClearDepth(1.0)
        glDepthFunc(GL_LEQUAL)
        glShadeModel(GL_SMOOTH)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glDisable(GL_COLOR_MATERIAL)

        glEnable(GL_CULL_FACE)  # Enable backface culling
        glCullFace(GL_BACK)     # Cull back faces
        glEnable(GL_MULTISAMPLE)  # Enable anti-aliasing
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        
        # Set up primary light - blueish-white like old version
        glLightfv(GL_LIGHT0, GL_POSITION, [1, 1, 1, 0])  # Directional
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.2, 1.0])  # Blueish ambient
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.9, 1.0])  # Blueish-white diffuse
        glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.5, 0.7, 1.0])  # Blueish specular
        
        # Add secondary light like old version
        glEnable(GL_LIGHT1)
        glLightfv(GL_LIGHT1, GL_POSITION, [-1, -1, 1, 0])  # From opposite direction
        glLightfv(GL_LIGHT1, GL_AMBIENT, [0.05, 0.05, 0.1, 1.0])
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.4, 0.4, 0.6, 1.0])
        glLightfv(GL_LIGHT1, GL_SPECULAR, [0.3, 0.3, 0.5, 1.0])
        
        # Initialize user-added lights (will start from GL_LIGHT2)
        for i, (name, light) in enumerate(self.lights.items(), start=2):
            if i >= 8:  # Only support up to GL_LIGHT7
                break
            light.setup_light(GL_LIGHT0 + i)
        
        glEnable(GL_NORMALIZE)
        self.star_background = StarBackground(1000)

        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)

    def paintGL(self):
        # Clear the screen and depth buffer
        glClearColor(0.05, 0.05, 0.15, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Set up projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = self.width()/float(self.height() or 1)
        gluPerspective(45.0, aspect, 0.1, 200.0)
        
        # Set up view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        # Calculate camera direction vectors
        front_x = math.cos(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))
        front_y = math.sin(math.radians(self.camera_pitch))
        front_z = math.sin(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))
        front = [front_x, front_y, front_z]
        
        # Normalize front vector
        front_mag = math.sqrt(front_x**2 + front_y**2 + front_z**2)
        front = [x/front_mag for x in front]
        
        # Calculate target position (center of view)
        center = [
            self.camera_pos[0] + front[0],
            self.camera_pos[1] + front[1],
            self.camera_pos[2] + front[2]
        ]
        
        # Set up the view matrix
        gluLookAt(
            self.camera_pos[0], self.camera_pos[1], self.camera_pos[2],  # Camera position
            center[0], center[1], center[2],  # Look at point
            0, 1, 0                          # Up vector
        )
        
        # Draw star background (after setting up view matrix)
        if self.star_background:
            glPushMatrix()
            glLoadIdentity()
            
            # Apply inverse of camera rotation to keep stars fixed relative to world
            # (but still allow model_angle rotation)
            glRotatef(self.camera_pitch, 1, 0, 0)
            glRotatef(self.camera_yaw, 0, 1, 0)
            glRotatef(self.angle, 0, 1, 0)  # Slow rotation if auto_rotate is enabled
            
            self.star_background.draw(
                self.camera_pos,
                self.camera_pitch,
                self.camera_yaw,
                self.angle
            )
            glPopMatrix()
        
        # Lighting setup
        if self.use_lighting:
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_LIGHT1)
            
            # Update all user lights
            for light in self.lights.values():
                if light.enabled:
                    light.update_light()
                    if light.light_id:
                        glEnable(light.light_id)
                else:
                    if light.light_id:
                        glDisable(light.light_id)
        else:
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
            glDisable(GL_LIGHT1)
            for light in self.lights.values():
                if light.light_id:
                    glDisable(light.light_id)
        
        # Draw all models
        for name, model in self.models.items():
            model.draw()
            
            # Draw gizmo for selected model if we're in models tab
            if (self.parent_window.current_model == name and 
                self.parent_window.tab_widget.currentIndex() == 1):  # 1 = Models tab index
                self.draw_position_gizmo(model)

        # Draw all light markers
        for light in self.lights.values():
            light.draw_marker()
        
        # Draw coordinate axes for reference (optional)
        if hasattr(self, 'show_axes') and self.show_axes:
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            # X axis (red)
            glColor3f(1, 0, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(1, 0, 0)
            # Y axis (green)
            glColor3f(0, 1, 0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 1, 0)
            # Z axis (blue)
            glColor3f(0, 0, 1)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 1)
            glEnd()
            glPopAttrib()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / float(h or 1), 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)

    def auto_rotate_update(self):
        if self.auto_rotate:
            # Only modify yaw (horizontal rotation) - leave pitch unchanged
            self.camera_yaw += 0.5  # Adjust this value to control rotation speed
            self.update()  # Force redraw
        
        # Handle movement keys as before
        if self.forward or self.backward or self.left or self.right or self.up or self.down:
            self.update_camera_position()
            self.update()

    def add_model(self, name, filename=""):
        if name in self.models:
            return False
        model = ModelObject(filename)
        model.filename = filename  # Ensure filename is set
        self.models[name] = model
        self.update()
        return True

    def remove_model(self, name):
        if name in self.models:
            del self.models[name]
            self.update()
            return True
        return False

    def add_light(self, name):
        if name in self.lights:
            return False
        
        # Find next available light ID (GL_LIGHT1 to GL_LIGHT7)
        for i in range(1, 8):
            light_id = GL_LIGHT0 + i
            if not any(light.light_id == light_id for light in self.lights.values()):
                light = LightObject()
                light.setup_light(light_id)
                self.lights[name] = light
                self.update()
                return True
        return False

    def remove_light(self, name):
        if name in self.lights:
            if self.lights[name].light_id:
                glDisable(self.lights[name].light_id)
            del self.lights[name]
            self.update()
            return True
        return False

    def mousePressEvent(self, event):
        if (event.button() == Qt.LeftButton and 
            self.parent_window.tab_widget.currentIndex() == 1 and 
            self.parent_window.current_model):
            
            # Get mouse position
            pos = event.pos()
            x, y = pos.x(), pos.y()
            
            # Simple check if we clicked near the gizmo
            # (This is a simplified version - you might want to implement proper raycasting)
            self.dragging_axis = None
            model = self.models.get(self.parent_window.current_model)
            if model:
                # You'll need to implement proper axis selection here
                # For now we'll just set a flag
                self.dragging_model = model
                self.last_mouse_pos = pos
                return

        if event.button() == Qt.RightButton:
            self.rotating = True
            self.last_pos = event.pos()
        elif event.button() == Qt.MiddleButton:
            self.reset_view()

    def mouseReleaseEvent(self, event):
        if hasattr(self, 'dragging_model'):
            del self.dragging_model

        if event.button() == Qt.RightButton:
            self.rotating = False

    def mouseMoveEvent(self, event):
        if hasattr(self, 'dragging_model') and self.dragging_model:
            # Calculate mouse delta
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            
            # Simple movement - you can refine this to move along specific axes
            move_speed = 0.01
            self.dragging_model.position[0] += delta.x() * move_speed
            self.dragging_model.position[1] -= delta.y() * move_speed  # Inverted Y
            
            # Update the UI spin boxes
            self.parent_window.model_x_spin.setValue(self.dragging_model.position[0])
            self.parent_window.model_y_spin.setValue(self.dragging_model.position[1])
            self.parent_window.model_z_spin.setValue(self.dragging_model.position[2])
            
            self.update()
            return

        if self.rotating and event.buttons() & Qt.RightButton:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            
            self.camera_yaw += dx * self.camera_sensitivity
            self.camera_pitch -= dy * self.camera_sensitivity
            
            # Constrain pitch to avoid gimbal lock
            self.camera_pitch = max(-89.0, min(89.0, self.camera_pitch))
            
            self.last_pos = event.pos()
            self.update()

    def reset_view(self):
        self.camera_pos = [0.0, 0.0, 5.0]  # Start slightly back from origin
        self.camera_pitch = 0.0
        self.camera_yaw = -90.0  # Looking along -Z axis
        self.update()

    def wheelEvent(self, event):
        zoom_factor = 0.1
        if event.angleDelta().y() > 0:
            self.camera_distance *= (1 - zoom_factor)
        else:
            self.camera_distance *= (1 + zoom_factor)
        self.update()

    def add_model(self, name, filename=""):
        if name in self.models:
            return False
        self.models[name] = ModelObject(filename)
        self.update()
        return True

    def remove_model(self, name):
        if name in self.models:
            del self.models[name]
            self.update()
            return True
        return False

    def add_light(self, name):
        if name in self.lights:
            return False
        
        # Find the next available light ID (GL_LIGHT0 to GL_LIGHT7)
        for i in range(0, 8):  # GL_LIGHT0 to GL_LIGHT7
            light_id = GL_LIGHT0 + i
            if light_id > GL_LIGHT7:  # Ensure we don't exceed maximum lights
                return False
                
            if not any(light.light_id == light_id for light in self.lights.values()):
                light = LightObject()
                self.lights[name] = light
                # We'll setup the light in initializeGL
                self.update()
                return True
        
        return False
    
    def remove_light(self, name):
        if name in self.lights:
            if self.lights[name].light_id:
                glDisable(self.lights[name].light_id)
            del self.lights[name]
            self.update()
            return True
        return False

    def save_scene(self, filename):
        try:
            scene_data = {
                'models': {},
                'lights': {},
                'camera': {
                    'pos': self.camera_pos,
                    'rot_x': self.camera_rot_x,
                    'rot_y': self.camera_rot_y,
                    'distance': self.camera_distance
                },
                'settings': {
                    'lighting': self.use_lighting,
                    'auto_rotate': self.auto_rotate
                }
            }
            
            for display_name, model in self.models.items():
                # Use the display name as-is for the key, don't try to extract base name
                internal_name = display_name
                
                model_path = model.filename if hasattr(model, 'filename') else ""
                
                # Convert to relative path if in same directory
                if model_path and os.path.isabs(model_path):
                    try:
                        scene_dir = os.path.dirname(os.path.abspath(filename))
                        rel_path = os.path.relpath(model_path, scene_dir)
                        if not rel_path.startswith('..'):
                            model_path = rel_path
                    except:
                        pass
                
                scene_data['models'][internal_name] = {
                    'filename': model_path.replace('\\', '/'),  # Use forward slashes
                    'position': model.position,
                    'rotation': model.rotation,
                    'scale': model.scale,
                    'visible': model.visible,
                    'wireframe': model.wireframe,
                    'display_name': display_name  # Store the display name for reconstruction
                }
            
            for name, light in self.lights.items():
                scene_data['lights'][name] = {
                    'position': light.position,
                    'color': light.color,
                    'brightness': light.brightness,
                    'range': light.range,
                    'enabled': light.enabled,
                    'show_marker': light.show_marker
                }
            
            with open(filename, 'w') as f:
                json.dump(scene_data, f, indent=4)
            
            print(f"Scene saved successfully to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving scene: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load_scene(self, filename):
        try:
            with open(filename, 'r') as f:
                scene_data = json.load(f)
            
            # Clear current scene
            self.models.clear()
            self.lights.clear()
            
            # Load models
            for internal_name, model_data in scene_data.get('models', {}).items():
                model_path = model_data.get('filename', '')
                display_name = model_data.get('display_name', internal_name)
                
                # Convert relative path to absolute if needed
                if model_path and not os.path.isabs(model_path):
                    model_path = os.path.join(os.path.dirname(os.path.abspath(filename)), model_path)
                
                # Create model (even if file doesn't exist - it might be loaded later)
                model = ModelObject(model_path if os.path.exists(model_path) else "")
                self.models[display_name] = model
                
                # Set properties
                model.position = model_data.get('position', [0.0, 0.0, 0.0])
                model.rotation = model_data.get('rotation', [0.0, 0.0, 0.0])
                model.scale = model_data.get('scale', 1.0)
                model.visible = model_data.get('visible', True)
                model.wireframe = model_data.get('wireframe', False)
                
                # Try to load texture if model file exists
                if model_path and os.path.exists(model_path):
                    model.try_load_texture()
            
            # Rest of the loading code remains the same...
            # Load lights
            for name, light_data in scene_data.get('lights', {}).items():
                light = LightObject()
                self.lights[name] = light
                light.position = light_data.get('position', [1.0, 1.0, 1.0])
                light.color = light_data.get('color', [0.9, 0.8, 0.6])
                light.brightness = light_data.get('brightness', 1.0)
                light.range = light_data.get('range', 1.0)
                light.enabled = light_data.get('enabled', True)
                light.show_marker = light_data.get('show_marker', True)
            
            # Load camera
            camera = scene_data.get('camera', {})
            self.camera_pos = camera.get('pos', [0.0, 0.0, 0.0])
            self.camera_rot_x = camera.get('rot_x', 30.0)
            self.camera_rot_y = camera.get('rot_y', 30.0)
            self.camera_distance = camera.get('distance', 5.0)
            
            # Load settings
            settings = scene_data.get('settings', {})
            self.use_lighting = settings.get('lighting', True)
            self.auto_rotate = settings.get('auto_rotate', True)
            
            # Reinitialize lights
            self.initializeGL()
            self.update()
            
            return True
        except Exception as e:
            print(f"Error loading scene: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
    def get_relative_path(self, abs_path, scene_path):
        """Convert absolute path to relative path if in same directory or subdirectory"""
        if not abs_path or not scene_path:
            return abs_path
        
        try:
            scene_dir = os.path.dirname(os.path.abspath(scene_path))
            abs_path = os.path.abspath(abs_path)
            
            # Try to get relative path
            rel_path = os.path.relpath(abs_path, scene_dir)
            
            # Only use relative path if it doesn't go up directories (starts with ..)
            if not rel_path.startswith('..'):
                return rel_path.replace('\\', '/')  # Use forward slashes for consistency
            return abs_path.replace('\\', '/')
        except:
            return abs_path.replace('\\', '/')

    def get_absolute_path(self, rel_path, scene_path):
        """Convert relative path to absolute path based on scene file location"""
        if not rel_path or not scene_path:
            return rel_path
        
        try:
            scene_dir = os.path.dirname(os.path.abspath(scene_path))
            abs_path = os.path.join(scene_dir, rel_path)
            return os.path.normpath(abs_path)
        except:
            return rel_path

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Z:  # Z - forward
            self.forward = True
        elif key == Qt.Key_S:  # S - backward
            self.backward = True
        elif key == Qt.Key_Q:  # Q - left
            self.left = True
        elif key == Qt.Key_D:  # D - right
            self.right = True
        elif key == Qt.Key_W:  # W - up (for QWERTY users)
            self.up = True
        elif key == Qt.Key_A:  # A - down (for QWERTY users)
            self.down = True
        elif key == Qt.Key_Space:  # Space - up
            self.up = True
        elif key == Qt.Key_Shift:  # Shift - down
            self.down = True

    def keyReleaseEvent(self, event):
        key = event.key()
        if key == Qt.Key_Z:
            self.forward = False
        elif key == Qt.Key_S:
            self.backward = False
        elif key == Qt.Key_Q:
            self.left = False
        elif key == Qt.Key_D:
            self.right = False
        elif key == Qt.Key_W:
            self.up = False
        elif key == Qt.Key_A:
            self.down = False
        elif key == Qt.Key_Space:
            self.up = False
        elif key == Qt.Key_Shift:
            self.down = False

    def update_camera_position(self):
        # Calculate camera front vector
        front_x = math.cos(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))
        front_y = math.sin(math.radians(self.camera_pitch))
        front_z = math.sin(math.radians(self.camera_yaw)) * math.cos(math.radians(self.camera_pitch))
        front = [front_x, front_y, front_z]
        
        # Normalize front vector
        front_mag = math.sqrt(front_x**2 + front_y**2 + front_z**2)
        front = [x/front_mag for x in front]
        
        # Calculate right vector (perpendicular to front and world up)
        right = np.cross(front, [0, 1, 0])  # Cross product with world up vector
        right_mag = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
        right = [x/right_mag for x in right]
        
        move_speed = self.camera_speed
        
        # Movement - now properly using front and right vectors
        if self.forward:  # Z key
            self.camera_pos[0] += front[0] * move_speed
            self.camera_pos[1] += front[1] * move_speed
            self.camera_pos[2] += front[2] * move_speed
        if self.backward:  # S key
            self.camera_pos[0] -= front[0] * move_speed
            self.camera_pos[1] -= front[1] * move_speed
            self.camera_pos[2] -= front[2] * move_speed
        if self.left:  # Q key - strafe left
            self.camera_pos[0] -= right[0] * move_speed
            self.camera_pos[2] -= right[2] * move_speed
        if self.right:  # D key - strafe right
            self.camera_pos[0] += right[0] * move_speed
            self.camera_pos[2] += right[2] * move_speed
        if self.up:  # Space key
            self.camera_pos[1] += move_speed
        if self.down:  # Shift key
            self.camera_pos[1] -= move_speed

    def draw_position_gizmo(self, model):
        if not model or not model.visible:
            return
        
        glPushMatrix()
        glTranslatef(*model.position)
        
        # Set line width
        glLineWidth(3.0)
        
        # Disable lighting for the gizmo
        glDisable(GL_LIGHTING)
        
        # X axis (Red)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)  # Red
        glVertex3f(0, 0, 0)
        glVertex3f(1, 0, 0)
        glEnd()
        
        # Y axis (Green)
        glBegin(GL_LINES)
        glColor3f(0, 1, 0)  # Green
        glVertex3f(0, 0, 0)
        glVertex3f(0, 1, 0)
        glEnd()
        
        # Z axis (Blue)
        glBegin(GL_LINES)
        glColor3f(0, 0, 1)  # Blue
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 1)
        glEnd()
        
        # Restore settings
        glEnable(GL_LIGHTING)
        glLineWidth(1.0)
        glPopMatrix()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('3D Scene Editor')
        self.setGeometry(100, 100, 1200, 800)
        
        self.opengl_widget = OpenGLWidget(self)
        self.current_model = None
        self.current_light = None
        self.light_counter = 1
        self.model_counter = 1
        
        self.tab_widget = QTabWidget()
        self.scene_tab = QWidget()
        self.models_tab = QWidget()
        self.lights_tab = QWidget()
        
        self.tab_widget.addTab(self.scene_tab, "Scene")
        self.tab_widget.addTab(self.models_tab, "Models")
        self.tab_widget.addTab(self.lights_tab, "Lights")
        
        self.setup_scene_tab()
        self.setup_models_tab()
        self.setup_lights_tab()
        
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.opengl_widget, 3)
        
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.tab_widget)
        
        save_load_group = QGroupBox("Scene Management")
        save_load_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Scene")
        self.load_button = QPushButton("Load Scene")
        
        save_load_layout.addWidget(self.save_button)
        save_load_layout.addWidget(self.load_button)
        save_load_group.setLayout(save_load_layout)
        right_panel.addWidget(save_load_group)
        
        right_panel_widget = QWidget()
        right_panel_widget.setLayout(right_panel)
        right_panel_widget.setMaximumWidth(400)
        
        main_layout.addWidget(right_panel_widget, 1)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.save_button.clicked.connect(self.save_scene)
        self.load_button.clicked.connect(self.load_scene)
        
        # Add default light
        self.opengl_widget.add_light("Light 1")
        self.light_combo.addItem("Light 1")
        self.light_combo.setCurrentIndex(0)
        self.current_light = "Light 1"
        self.update_light_controls()

    def update_light_controls(self):
        if not hasattr(self, 'light_combo') or not hasattr(self, 'current_light'):
            return
            
        if self.light_combo.count() > 0 and self.current_light:
            light = self.opengl_widget.lights[self.current_light]
            self.light_x_spin.setValue(light.position[0])
            self.light_y_spin.setValue(light.position[1])
            self.light_z_spin.setValue(light.position[2])
            self.light_brightness_slider.setValue(int(light.brightness * 100))
            self.light_range_spin.setValue(light.range)
            self.light_enable_check.setChecked(light.enabled)
            self.light_marker_check.setChecked(light.show_marker)
            
            # Update color button
            color = QColor()
            color.setRgbF(*light.color)
            self.light_color_button.setStyleSheet(f"background-color: {color.name()}")

    def setup_scene_tab(self):
        layout = QVBoxLayout()
        
        camera_group = QGroupBox("Camera")
        camera_layout = QFormLayout()
        
        self.cam_x_spin = QDoubleSpinBox()
        self.cam_x_spin.setRange(-100, 100)
        camera_layout.addRow("X Pos:", self.cam_x_spin)
        
        self.cam_y_spin = QDoubleSpinBox()
        self.cam_y_spin.setRange(-100, 100)
        camera_layout.addRow("Y Pos:", self.cam_y_spin)
        
        self.cam_z_spin = QDoubleSpinBox()
        self.cam_z_spin.setRange(-100, 100)
        camera_layout.addRow("Z Pos:", self.cam_z_spin)
        
        self.cam_dist_spin = QDoubleSpinBox()
        self.cam_dist_spin.setRange(0.1, 100)
        self.cam_dist_spin.setValue(5.0)
        camera_layout.addRow("Distance:", self.cam_dist_spin)
        
        self.cam_rot_x_spin = QDoubleSpinBox()
        self.cam_rot_x_spin.setRange(-180, 180)
        self.cam_rot_x_spin.setValue(30.0)
        camera_layout.addRow("X Rotation:", self.cam_rot_x_spin)
        
        self.cam_rot_y_spin = QDoubleSpinBox()
        self.cam_rot_y_spin.setRange(-180, 180)
        self.cam_rot_y_spin.setValue(30.0)
        camera_layout.addRow("Y Rotation:", self.cam_rot_y_spin)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        options_group = QGroupBox("Scene Options")
        options_layout = QVBoxLayout()
        
        self.lighting_check = QCheckBox("Enable Lighting")
        self.lighting_check.setChecked(True)
        options_layout.addWidget(self.lighting_check)
        
        self.auto_rotate_check = QCheckBox("Auto Rotate")
        self.auto_rotate_check.setChecked(True)
        options_layout.addWidget(self.auto_rotate_check)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        self.scene_tab.setLayout(layout)
        
        self.cam_x_spin.valueChanged.connect(self.update_camera_position)
        self.cam_y_spin.valueChanged.connect(self.update_camera_position)
        self.cam_z_spin.valueChanged.connect(self.update_camera_position)
        self.cam_dist_spin.valueChanged.connect(self.update_camera_distance)
        self.cam_rot_x_spin.valueChanged.connect(self.update_camera_rotation)
        self.cam_rot_y_spin.valueChanged.connect(self.update_camera_rotation)
        self.lighting_check.stateChanged.connect(self.toggle_lighting)
        self.auto_rotate_check.stateChanged.connect(self.toggle_auto_rotate)

    def setup_models_tab(self):
        layout = QVBoxLayout()
        
        # Model selection group
        model_select_group = QGroupBox("Model Selection")
        model_select_layout = QHBoxLayout()
        
        # Replace the model combo box setup with this:
        self.model_list = QListWidget()
        self.model_list.setMinimumWidth(250)  # Wider to show full names
        self.model_list.setSelectionMode(QListWidget.SingleSelection)
        self.remove_model_button = QPushButton("Remove")
        self.duplicate_model_button = QPushButton("Duplicate")
        self.load_model_button = QPushButton("Load OBJ")

        model_select_layout.addWidget(self.model_list)
        model_select_layout.addWidget(self.remove_model_button)
        model_select_layout.addWidget(self.duplicate_model_button)
        model_select_layout.addWidget(self.load_model_button)

        model_select_group.setLayout(model_select_layout)
        layout.addWidget(model_select_group)
        
        # Model controls group (same controls as before)
        self.model_controls_group = QGroupBox("Model Controls")
        model_controls_layout = QFormLayout()
        
        self.model_x_spin = QDoubleSpinBox()
        self.model_x_spin.setRange(-100, 100)
        model_controls_layout.addRow("X Pos:", self.model_x_spin)
        
        self.model_y_spin = QDoubleSpinBox()
        self.model_y_spin.setRange(-100, 100)
        model_controls_layout.addRow("Y Pos:", self.model_y_spin)
        
        self.model_z_spin = QDoubleSpinBox()
        self.model_z_spin.setRange(-100, 100)
        model_controls_layout.addRow("Z Pos:", self.model_z_spin)
        
        self.model_rot_x_spin = QDoubleSpinBox()
        self.model_rot_x_spin.setRange(-180, 180)
        model_controls_layout.addRow("X Rot:", self.model_rot_x_spin)
        
        self.model_rot_y_spin = QDoubleSpinBox()
        self.model_rot_y_spin.setRange(-180, 180)
        model_controls_layout.addRow("Y Rot:", self.model_rot_y_spin)
        
        self.model_rot_z_spin = QDoubleSpinBox()
        self.model_rot_z_spin.setRange(-180, 180)
        model_controls_layout.addRow("Z Rot:", self.model_rot_z_spin)
        
        self.model_scale_slider = QSlider(Qt.Horizontal)
        self.model_scale_slider.setRange(1, 200)
        self.model_scale_slider.setValue(100)
        model_controls_layout.addRow("Scale:", self.model_scale_slider)
        
        self.model_visible_check = QCheckBox("Visible")
        self.model_visible_check.setChecked(True)
        model_controls_layout.addRow(self.model_visible_check)
        
        self.model_wireframe_check = QCheckBox("Wireframe")
        model_controls_layout.addRow(self.model_wireframe_check)
        
        self.model_controls_group.setLayout(model_controls_layout)
        layout.addWidget(self.model_controls_group)
        
        self.models_tab.setLayout(layout)
        
        # With these:
        self.model_list.itemSelectionChanged.connect(self.model_selected)
        self.remove_model_button.clicked.connect(self.remove_model)
        self.duplicate_model_button.clicked.connect(self.duplicate_model)
        self.load_model_button.clicked.connect(self.load_model_file)
        
        self.model_x_spin.valueChanged.connect(self.update_model_position)
        self.model_y_spin.valueChanged.connect(self.update_model_position)
        self.model_z_spin.valueChanged.connect(self.update_model_position)
        self.model_rot_x_spin.valueChanged.connect(self.update_model_rotation)
        self.model_rot_y_spin.valueChanged.connect(self.update_model_rotation)
        self.model_rot_z_spin.valueChanged.connect(self.update_model_rotation)
        self.model_scale_slider.valueChanged.connect(self.update_model_scale)
        self.model_visible_check.stateChanged.connect(self.toggle_model_visibility)
        self.model_wireframe_check.stateChanged.connect(self.toggle_model_wireframe)

        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        # This will help the OpenGL widget know which tab is active
        pass  # The widget will check tab_widget.currentIndex()

    def add_model(self):
        name = f"Model {self.model_counter}"
        while name in self.opengl_widget.models:
            self.model_counter += 1
            name = f"Model {self.model_counter}"
        
        if self.opengl_widget.add_model(name):
            self.model_combo.addItem(name)
            self.model_combo.setCurrentText(name)
            self.model_counter += 1

    def remove_model(self):
        if not self.current_model:
            return
            
        if self.opengl_widget.remove_model(self.current_model):
            index = self.model_combo.currentIndex()
            self.model_combo.removeItem(index)
            
            if self.model_combo.count() > 0:
                self.model_combo.setCurrentIndex(0 if index == 0 else index - 1)
                self.model_selected(self.model_combo.currentText())
            else:
                self.current_model = None
                self.model_controls_group.setEnabled(False)

    def duplicate_model(self):
        if not self.current_model:
            return
            
        # Get the base name without any numbering
        base_name = self.current_model
        if ' (' in base_name:
            base_name = base_name.split(' (')[0]
        
        # Find all existing models with this base name
        existing_numbers = []
        for name in self.opengl_widget.models:
            if name.startswith(base_name + ' ('):
                try:
                    num = int(name.split(' (')[1].split(')')[0])
                    existing_numbers.append(num)
                except:
                    pass
        
        # Determine the next available number
        new_number = 1
        if existing_numbers:
            new_number = max(existing_numbers) + 1
        
        # Create the new display name
        new_display_name = f"{base_name} ({new_number})"
        
        # Use the base name as the internal name (for JSON key)
        internal_name = base_name
        
        # Duplicate the model
        if self.opengl_widget.add_model(new_display_name, self.opengl_widget.models[self.current_model].filename):
            # Set the internal name
            self.opengl_widget.models[new_display_name].internal_name = internal_name
            
            # Copy all properties
            new_model = self.opengl_widget.models[new_display_name]
            old_model = self.opengl_widget.models[self.current_model]
            
            new_model.position = old_model.position.copy()
            new_model.rotation = old_model.rotation.copy()
            new_model.scale = old_model.scale
            new_model.visible = old_model.visible
            new_model.wireframe = old_model.wireframe
            
            # Add to list and select
            self.model_list.addItem(new_display_name)
            self.model_list.setCurrentRow(self.model_list.count()-1)
            self.opengl_widget.update()

    def update_light_properties(self):
        if not self.current_light:
            return
            
        light = self.opengl_widget.lights[self.current_light]
        light.brightness = self.light_brightness_slider.value() / 100.0
        light.range = self.light_range_spin.value()
        self.opengl_widget.update()

    def choose_light_color(self):
        if not self.current_light:
            return
            
        light = self.opengl_widget.lights[self.current_light]
        color = QColorDialog.getColor()
        if color.isValid():
            light.color = [color.redF(), color.greenF(), color.blueF()]
            self.light_color_button.setStyleSheet(f"background-color: {color.name()}")
            self.opengl_widget.update()

    def add_light(self):
        name = f"Light {self.light_counter}"
        while name in self.opengl_widget.lights:
            self.light_counter += 1
            name = f"Light {self.light_counter}"
        
        if self.opengl_widget.add_light(name):
            self.light_combo.addItem(name)
            self.light_combo.setCurrentText(name)
            self.light_counter += 1

    def remove_light(self):
        if not self.current_light:
            return
            
        if self.opengl_widget.remove_light(self.current_light):
            index = self.light_combo.currentIndex()
            self.light_combo.removeItem(index)
            
            if self.light_combo.count() > 0:
                self.light_combo.setCurrentIndex(0 if index == 0 else index - 1)
                self.light_selected(self.light_combo.currentText())
            else:
                self.current_light = None
                self.light_controls_group.setEnabled(False)

    def update_light_position(self):
        if not self.current_light:
            return
            
        light = self.opengl_widget.lights[self.current_light]
        light.position = [
            self.light_x_spin.value(),
            self.light_y_spin.value(),
            self.light_z_spin.value()
        ]
        self.opengl_widget.update()

    def toggle_light_enabled(self, state):
        if not self.current_light:
            return
            
        light = self.opengl_widget.lights[self.current_light]
        light.enabled = state == Qt.Checked
        
        # Force OpenGL to update the light state
        if light.light_id:
            if light.enabled:
                glEnable(light.light_id)
            else:
                glDisable(light.light_id)
        
        self.opengl_widget.update()

    def toggle_light_marker(self, state):
        if not self.current_light:
            return
            
        self.opengl_widget.lights[self.current_light].show_marker = state == Qt.Checked
        self.opengl_widget.update()

    def setup_lights_tab(self):
        layout = QVBoxLayout()
        
        light_select_group = QGroupBox("Light Selection")
        light_select_layout = QHBoxLayout()
        
        self.light_combo = QComboBox()
        self.add_light_button = QPushButton("+")
        self.remove_light_button = QPushButton("-")
        
        light_select_layout.addWidget(self.light_combo)
        light_select_layout.addWidget(self.add_light_button)
        light_select_layout.addWidget(self.remove_light_button)
        light_select_group.setLayout(light_select_layout)
        layout.addWidget(light_select_group)
        
        self.light_controls_group = QGroupBox("Light Controls")
        light_controls_layout = QFormLayout()
        
        # Remove range restrictions from these spin boxes
        self.light_x_spin = QDoubleSpinBox()
        self.light_x_spin.setRange(-999999, 999999)  # Large range instead of -10 to 10
        self.light_x_spin.setValue(1.0)
        light_controls_layout.addRow("X Pos:", self.light_x_spin)
        
        self.light_y_spin = QDoubleSpinBox()
        self.light_y_spin.setRange(-999999, 999999)
        self.light_y_spin.setValue(1.0)
        light_controls_layout.addRow("Y Pos:", self.light_y_spin)
        
        self.light_z_spin = QDoubleSpinBox()
        self.light_z_spin.setRange(-999999, 999999)
        self.light_z_spin.setValue(1.0)
        light_controls_layout.addRow("Z Pos:", self.light_z_spin)
        
        # Rest of the method remains the same...
        self.light_brightness_slider = QSlider(Qt.Horizontal)
        self.light_brightness_slider.setRange(10, 500)
        self.light_brightness_slider.setValue(100)
        light_controls_layout.addRow("Brightness:", self.light_brightness_slider)
        
        self.light_range_spin = QDoubleSpinBox()
        self.light_range_spin.setRange(0.1, 20.0)
        self.light_range_spin.setValue(1.0)
        self.light_range_spin.setSingleStep(0.1)
        light_controls_layout.addRow("Range:", self.light_range_spin)
        
        self.light_color_button = QPushButton("Choose Color")
        self.light_color_button.setStyleSheet("background-color: rgb(230, 200, 150)")
        light_controls_layout.addRow("Color:", self.light_color_button)
        
        self.light_enable_check = QCheckBox("Enabled")
        self.light_enable_check.setChecked(True)
        light_controls_layout.addRow(self.light_enable_check)
        
        self.light_marker_check = QCheckBox("Show Marker")
        self.light_marker_check.setChecked(True)
        light_controls_layout.addRow(self.light_marker_check)
        
        self.light_controls_group.setLayout(light_controls_layout)
        layout.addWidget(self.light_controls_group)
        
        self.lights_tab.setLayout(layout)
        
        self.light_combo.currentTextChanged.connect(self.light_selected)
        self.add_light_button.clicked.connect(self.add_light)
        self.remove_light_button.clicked.connect(self.remove_light)
        
        self.light_x_spin.valueChanged.connect(self.update_light_position)
        self.light_y_spin.valueChanged.connect(self.update_light_position)
        self.light_z_spin.valueChanged.connect(self.update_light_position)
        self.light_brightness_slider.valueChanged.connect(self.update_light_properties)
        self.light_range_spin.valueChanged.connect(self.update_light_properties)
        self.light_color_button.clicked.connect(self.choose_light_color)
        self.light_enable_check.stateChanged.connect(self.toggle_light_enabled)
        self.light_marker_check.stateChanged.connect(self.toggle_light_marker)

    def update_camera_controls(self):
        self.cam_x_spin.setValue(self.opengl_widget.camera_pos[0])
        self.cam_y_spin.setValue(self.opengl_widget.camera_pos[1])
        self.cam_z_spin.setValue(self.opengl_widget.camera_pos[2])
        self.cam_dist_spin.setValue(self.opengl_widget.camera_distance)
        self.cam_rot_x_spin.setValue(self.opengl_widget.camera_rot_x)
        self.cam_rot_y_spin.setValue(self.opengl_widget.camera_rot_y)

    def update_camera_position(self):
        self.opengl_widget.camera_pos = [
            self.cam_x_spin.value(),
            self.cam_y_spin.value(),
            self.cam_z_spin.value()
        ]
        self.opengl_widget.update()

    def update_camera_rotation(self):
        self.opengl_widget.camera_rot_x = self.cam_rot_x_spin.value()
        self.opengl_widget.camera_rot_y = self.cam_rot_y_spin.value()
        self.opengl_widget.update()

    def update_camera_distance(self):
        self.opengl_widget.camera_distance = self.cam_dist_spin.value()
        self.opengl_widget.update()

    def toggle_lighting(self, state):
        self.opengl_widget.use_lighting = state == Qt.Checked
        self.opengl_widget.update()

    def toggle_auto_rotate(self, state):
        self.opengl_widget.auto_rotate = state == Qt.Checked

    def model_selected(self):
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            self.current_model = None
            self.model_controls_group.setEnabled(False)
            return
            
        name = selected_items[0].text()
        if not name or name not in self.opengl_widget.models:
            self.current_model = None
            self.model_controls_group.setEnabled(False)
            return
            
        # Block signals while we update the controls
        self.model_x_spin.blockSignals(True)
        self.model_y_spin.blockSignals(True)
        self.model_z_spin.blockSignals(True)
        self.model_rot_x_spin.blockSignals(True)
        self.model_rot_y_spin.blockSignals(True)
        self.model_rot_z_spin.blockSignals(True)
        self.model_scale_slider.blockSignals(True)
        self.model_visible_check.blockSignals(True)
        self.model_wireframe_check.blockSignals(True)
        
        model = self.opengl_widget.models[name]
        self.current_model = name
        
        self.model_x_spin.setValue(model.position[0])
        self.model_y_spin.setValue(model.position[1])
        self.model_z_spin.setValue(model.position[2])
        self.model_rot_x_spin.setValue(model.rotation[0])
        self.model_rot_y_spin.setValue(model.rotation[1])
        self.model_rot_z_spin.setValue(model.rotation[2])
        self.model_scale_slider.setValue(int(model.scale * 100))
        self.model_visible_check.setChecked(model.visible)
        self.model_wireframe_check.setChecked(model.wireframe)
        
        # Unblock signals after updating
        self.model_x_spin.blockSignals(False)
        self.model_y_spin.blockSignals(False)
        self.model_z_spin.blockSignals(False)
        self.model_rot_x_spin.blockSignals(False)
        self.model_rot_y_spin.blockSignals(False)
        self.model_rot_z_spin.blockSignals(False)
        self.model_scale_slider.blockSignals(False)
        self.model_visible_check.blockSignals(False)
        self.model_wireframe_check.blockSignals(False)
        
        self.model_controls_group.setEnabled(True)

    def add_model(self, name=None, filename=""):
        if name is None:
            base_name = os.path.splitext(os.path.basename(filename))[0] if filename else "Model"
            
            # Find all existing models with this base name
            existing_numbers = []
            for existing_name in self.opengl_widget.models:
                if existing_name.startswith(base_name + ' ('):
                    try:
                        num = int(existing_name.split(' (')[1].split(')')[0])
                        existing_numbers.append(num)
                    except:
                        pass
            
            # Determine the next available number
            new_number = 1
            if existing_numbers:
                new_number = max(existing_numbers) + 1
            
            display_name = f"{base_name} ({new_number})"
        else:
            display_name = name
        
        if self.opengl_widget.add_model(display_name, filename):
            # Set internal name to base name without numbering
            internal_name = display_name.split(' (')[0]
            self.opengl_widget.models[display_name].internal_name = internal_name
            
            self.model_list.addItem(display_name)
            self.model_list.setCurrentRow(self.model_list.count()-1)
            return True
        return False

    def remove_model(self):
        selected_items = self.model_list.selectedItems()
        if not selected_items:
            return
            
        name = selected_items[0].text()
        if not name or name not in self.opengl_widget.models:
            return
            
        if self.opengl_widget.remove_model(name):
            row = self.model_list.row(selected_items[0])
            self.model_list.takeItem(row)
            
            if self.model_list.count() > 0:
                self.model_list.setCurrentRow(0 if row == 0 else row - 1)
            else:
                self.current_model = None
                self.model_controls_group.setEnabled(False)

    def load_model_file(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open OBJ File", "", "OBJ Files (*.obj);;All Files (*)", options=options)
        
        if filename:
            name = os.path.splitext(os.path.basename(filename))[0]
            if name in self.opengl_widget.models:
                name = f"{name}_{self.model_counter}"
                self.model_counter += 1
            
            if self.add_model(name, filename):
                pass  # Already handled in add_model

    def update_model_position(self):
        if not self.current_model:
            return
            
        model = self.opengl_widget.models[self.current_model]
        model.position = [
            self.model_x_spin.value(),
            self.model_y_spin.value(),
            self.model_z_spin.value()
        ]
        self.opengl_widget.update()

    def update_model_rotation(self):
        if not self.current_model:
            return
            
        model = self.opengl_widget.models[self.current_model]
        model.rotation = [
            self.model_rot_x_spin.value(),
            self.model_rot_y_spin.value(),
            self.model_rot_z_spin.value()
        ]
        self.opengl_widget.update()

    def update_model_scale(self, value):
        if not self.current_model:
            return
            
        self.opengl_widget.models[self.current_model].scale = value / 100.0
        self.opengl_widget.update()

    def toggle_model_visibility(self, state):
        if not self.current_model:
            return
            
        self.opengl_widget.models[self.current_model].visible = state == Qt.Checked
        self.opengl_widget.update()

    def toggle_model_wireframe(self, state):
        if not self.current_model:
            return
            
        self.opengl_widget.models[self.current_model].wireframe = state == Qt.Checked
        self.opengl_widget.update()

    def light_selected(self, name):
        if not name or name not in self.opengl_widget.lights:
            self.current_light = None
            self.light_controls_group.setEnabled(False)
            return
            
        # Block signals while we update the controls
        self.light_x_spin.blockSignals(True)
        self.light_y_spin.blockSignals(True)
        self.light_z_spin.blockSignals(True)
        self.light_brightness_slider.blockSignals(True)
        self.light_range_spin.blockSignals(True)
        self.light_enable_check.blockSignals(True)
        self.light_marker_check.blockSignals(True)
        
        self.current_light = name
        light = self.opengl_widget.lights[name]
        
        self.light_x_spin.setValue(light.position[0])
        self.light_y_spin.setValue(light.position[1])
        self.light_z_spin.setValue(light.position[2])
        self.light_brightness_slider.setValue(int(light.brightness * 100))
        self.light_range_spin.setValue(light.range)
        self.light_enable_check.setChecked(light.enabled)
        self.light_marker_check.setChecked(light.show_marker)
        
        # Update color button style
        color = QColor()
        color.setRgbF(*light.color)
        self.light_color_button.setStyleSheet(f"background-color: {color.name()}")
        
        # Unblock signals after updating
        self.light_x_spin.blockSignals(False)
        self.light_y_spin.blockSignals(False)
        self.light_z_spin.blockSignals(False)
        self.light_brightness_slider.blockSignals(False)
        self.light_range_spin.blockSignals(False)
        self.light_enable_check.blockSignals(False)
        self.light_marker_check.blockSignals(False)
        
        self.light_controls_group.setEnabled(True)

    def save_scene(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self, "Save Scene", "", "JSON Files (*.json);;All Files (*)", options=options)
        
        if fileName:
            # Ensure the file has the .json extension
            if not fileName.lower().endswith('.json'):
                fileName += '.json'
            
            if self.opengl_widget.save_scene(fileName):
                print(f"Scene saved to {fileName}")
            else:
                print("Failed to save scene")

    def load_scene(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Load Scene", "", "JSON Files (*.json);;All Files (*)", options=options)
        
        if fileName:
            if self.opengl_widget.load_scene(fileName):
                # Update model list
                self.model_list.clear()
                for name in self.opengl_widget.models:
                    self.model_list.addItem(name)
                
                # Update light combo
                self.light_combo.clear()
                for name in self.opengl_widget.lights:
                    self.light_combo.addItem(name)
                
                # Update camera controls
                self.update_camera_controls()
                
                # Select first items if available
                if self.model_list.count() > 0:
                    self.model_list.setCurrentRow(0)
                    self.model_selected()
                
                if self.light_combo.count() > 0:
                    self.light_combo.setCurrentIndex(0)
                    self.light_selected(self.light_combo.currentText())
                
                # Update checkboxes
                self.lighting_check.setChecked(self.opengl_widget.use_lighting)
                self.auto_rotate_check.setChecked(self.opengl_widget.auto_rotate)
                
                print(f"Scene loaded from {fileName}")
                return True
            else:
                print("Failed to load scene")
                return False
        return False

if __name__ == '__main__':
    # Enable high DPI scaling if available
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec_())