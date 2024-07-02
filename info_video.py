import cv2

def get_video_info_and_save_middle_frame(video_path):
    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error al abrir el video.")
        return
    
    # Obtener FPS del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Obtener el número total de frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calcular el frame del medio
    middle_frame = total_frames // 2
    
    # Ir al frame del medio
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    # Leer el frame
    ret, frame = cap.read()
    
    if ret:
        # Guardar el frame como imagen
        cv2.imwrite("middle_frame.jpg", frame)
        print(f"Frame del medio guardado como 'middle_frame.jpg'")
    else:
        print("No se pudo leer el frame del medio")
    
    # Cerrar el video
    cap.release()
    
    # Imprimir información
    print(f"FPS del video: {fps}")
    print(f"Número total de frames: {total_frames}")
    print(f"Duración del video: {total_frames / fps:.2f} segundos")
    print(f"Un segundo equivale a {fps:.0f} frames")

# Uso de la función
video_path = "video.mp4"
get_video_info_and_save_middle_frame(video_path)