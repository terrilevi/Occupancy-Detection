import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Cargar el modelo YOLOv8
model = YOLO('yolov9m.pt')

# Definir los bounding boxes estáticos (sillas)
static_boxes = [
    [918, 4, 77, 98.17],
    [888, 100, 98.26, 112.77],
    [684, 127, 86.54, 124.08],
    [758, 182, 134.5, 163.7],
    [744, 64, 82.37, 93.84],
    [850, 0, 79.59, 44.83],
    [895, 176, 88.38, 137.63],
    [903, 71, 91.68, 85.83],
    [761, 40, 71.52, 68.92],
    [772, 0, 66.97, 73.47],
    [1063, 167, 153.45, 207.41],
    [1058, 106, 115.74, 133.94],
    [1042, 68, 101.43, 130.69],
    [1014, 24, 90.38, 118.34],
    [1105, 0, 98.83, 102.08],
    [1200, 48, 80.38, 96.23],
    [1175, 299, 105.08, 162.55],
    [881, 435, 236.84, 258.9],
    [1130, 449, 150.24, 242.72],
    [544, 366, 188.29, 200.06],
    [481, 168, 150.05, 169.17],
    [319, 196, 161.81, 186.82],
    [309, 124, 114.74, 142.69],
    [391, 82, 108.86, 125.04],
    [466, 34, 82.38, 110.33],
    [524, 18, 70.61, 91.2],
    [560, 0, 72.08, 66.2],
    [552, 87, 110.33, 154.46],
    [616, 62, 72.08, 120.63],
    [663, 35, 76.49, 113.27],
    [668, 0, 88.26, 75.02],
    [203, 119, 120.63, 139.75],
    [168, 54, 108.86, 86.79],
    [293, 65, 108.86, 138.28],
    [384, 43, 83.85, 104.44],
    [438, 12, 58.84, 79.44],
    [0, 147, 180.94, 158.87],
    [40, 81, 117.68, 111.8],
    [269, 28, 83.85, 76.49],
    [337, 7, 67.67, 64.73],
    [406, 0, 75.02, 48.54]
]

def calcular_midpoint(box):
    x, y, w, h = box
    return (x + w/2, y + h/2)

def calcular_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    
    return overlap_x * overlap_y > 0

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def proceso_frame(frame, static_boxes):
    # Detectar personas usando YOLO
    results = model(frame)
    
    # Extraer bounding boxes de personas detectadas
    person_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.cls == 0 and box.conf > 0.35: # Clase 0 es 'person' en COCO dataset
                x1, y1, x2, y2 = box.xyxy[0]
                person_boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
    
    # Calcular puntos medios
    person_midpoints = [calcular_midpoint(box) for box in person_boxes]
    static_midpoints = [calcular_midpoint(box) for box in static_boxes]
    
    # Inicializar diccionario para almacenar distancias
    distances = {i: {} for i in range(len(static_boxes))}
    
    # Calcular distancias para pares que se solapan
    for i, person_box in enumerate(person_boxes):
        for j, static_box in enumerate(static_boxes):
            if calcular_overlap(person_box, static_box):
                distance = calculate_distance(person_midpoints[i], static_midpoints[j])
                distances[j][i] = distance
    
    # Asignar personas a sillas
    assignments = {}
    assigned_people = set()
    
    # Ordenar las sillas por la distancia mínima a cualquier persona
    sorted_chairs = sorted(distances.items(), key=lambda x: min(x[1].values()) if x[1] else float('inf'))
    
    for chair_id, chair_distances in sorted_chairs:
        if chair_distances:
            # Encontrar la persona más cercana que aún no ha sido asignada
            available_people = [p for p in chair_distances.keys() if p not in assigned_people]
            if available_people:
                person_id = min(available_people, key=chair_distances.get)
                assignments[chair_id] = person_id
                assigned_people.add(person_id)
    
    return person_boxes, assignments

def process_video(video_path, start_time_str="11:35"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error al abrir el video.")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    frames_in_minutes = int(1 * 60 * fps)
    
    # Convertir la hora de inicio a un objeto datetime
    start_time = datetime.strptime(start_time_str, "%H:%M")
    
    # Para almacenar los datos para el gráfico
    time_points = []
    occupied_seats_count = []
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break
        
        person_boxes, assignments = proceso_frame(frame, static_boxes)

        # Dibujar bounding boxes y asignaciones
        occupied_seats = []
        free_seats = []
        for i, box in enumerate(static_boxes):
            x, y, w, h = box
            if i in assignments:  # Si la silla está asignada, colorearla de rojo
                color = (0, 0, 255)  # Rojo en BGR
                occupied_seats.append(i)
            else:
                color = (0, 255, 0)  # Verde en BGR
                free_seats.append(i)
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
        
        # Imprimir información de asientos en cada frame
        print(f"\nFrame {current_frame}:")
        print(f"Asientos ocupados ({len(occupied_seats)}): {occupied_seats}")
        print(f"Asientos libres ({len(free_seats)}): {free_seats}")
        print(f"Total de asientos: {len(static_boxes)}")
        
        # Calcular la hora actual del video
        current_second = current_frame / fps
        current_time = start_time + timedelta(seconds=current_second)
        
        # Almacenar datos para el gráfico
        time_points.append(current_time)
        occupied_seats_count.append(len(occupied_seats))
        
        for i, box in enumerate(person_boxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        for chair_id, person_id in assignments.items():
            chair_mid = calcular_midpoint(static_boxes[chair_id])
            person_mid = calcular_midpoint(person_boxes[person_id])
            cv2.line(frame, (int(chair_mid[0]), int(chair_mid[1])), 
                     (int(person_mid[0]), int(person_mid[1])), (0, 0, 255), 2)
        
        cv2.putText(frame, f"Tiempo: {current_time.strftime('%H:%M:%S')}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Video con Bounding Boxes", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_frame = min(current_frame + frames_in_minutes, total_frames - 1)
        elif key == ord('p'):
            current_frame = max(0, current_frame - frames_in_minutes)
    
    cap.release()
    cv2.destroyAllWindows()

    # Crear el gráfico
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, occupied_seats_count, marker='o', linestyle='-', markersize=4)
    plt.title('Ocupación de Asientos a lo largo del Tiempo')
    plt.xlabel('Hora del día')
    plt.ylabel('Número de Asientos Ocupados')
    plt.ylim(0, 42)  # Establecer el rango del eje Y de 0 a 42 para mostrar claramente hasta 41
    plt.yticks(range(1, 42))  # Establecer las marcas del eje Y de 1 a 41
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
    
    # Formatear el eje X para mostrar horas y minutos
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()  # Rotar y alinear las etiquetas de fecha
    
    plt.tight_layout()

    # Generar un nombre de archivo único basado en la fecha y hora actual
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocupacion_asientos_{timestamp}.png"
    
    # Guardar el gráfico como un archivo PNG
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado como: {filename}")

    # Mostrar el gráfico (opcional)
    plt.show()

# Uso del sistema
video_path = "video.mp4"
process_video(video_path, "11:35") 