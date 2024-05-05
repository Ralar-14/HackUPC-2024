from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
import random

app = Flask(__name__)
CORS(app)

#images = pd.read_csv('inditex_images.csv')
images = pd.read_csv('images_names.csv')
images_liked = images.combine_first(pd.DataFrame(0, index=images.index, columns=["scores"]))
images_liked.to_csv("liked_images.csv", index=False)
#image_embeddings = np.load("image_embeddings_1000.npy")
image_embeddings = np.load("emb_final.npy")

print(image_embeddings.shape)
print(images.shape)
color_embeddings = np.load("rgb_bueno.npy")
print(color_embeddings.shape)
not_voted = list(range(images.shape[0]))

def scoring(vote_type, image_id, color_value):
    color_value = int(color_value)
    image_embeddings_colored = np.concatenate([image_embeddings, color_embeddings*color_value], axis=1)
    index = faiss.IndexFlatL2(image_embeddings_colored.shape[1])
    index.add(image_embeddings_colored)
    D, I = index.search(image_embeddings_colored[image_id:image_id+1], 100)
    for sum, i in enumerate(I[0]):
        if i != image_id:
            if vote_type == "like":
                images_liked['scores'][i] += len(I[0]) - sum 
            else:
                images_liked['scores'][i] -= len(I[0]) - sum 
            
    images_liked['scores'][image_id] = -float('inf')
    images_liked.to_csv("liked_images.csv", index=False)

@app.route('/get-image')
def get_explore_image():
    # Devuelve una lista de imágenes para explorar
    image_data = [{"id": idx, "url": f"{images['path_to_image'][idx]}"} for idx in range(images.shape[0])]
    return jsonify(image_data)

@app.route('/to-rate-get-image')
def get_rating_image():
    # Devuelve una imagen aleatoria para calificar
    random_index = random.choice(not_voted)
    not_voted.remove(random_index)
    image_data = {
        "url": images['path_to_image'][random_index],
        "id": random_index  # Incluye el índice de la imagen
    }
    return jsonify(image_data)

@app.route('/reset')
def reset():
    images_liked = images.combine_first(pd.DataFrame(0, index=images.index, columns=["scores"]))
    images_liked.to_csv("liked_images.csv", index=False)

@app.route('/top-images')
def top_images():
    # Recargar el CSV para obtener datos actualizados
    images_liked = pd.read_csv("liked_images.csv")
    # Ordenar el DataFrame por scores de manera descendente y tomar los primeros 50 resultados
    top_images_data = images_liked.sort_values(by='scores', ascending=False).head(50)
    image_list = [{"id": int(index), "url": row['path_to_image'], "score": row['scores']} for index, row in top_images_data.iterrows()]
    return jsonify(image_list)


# Endpoint para registrar votos
@app.route('/vote', methods=['POST'])
def vote():
    data = request.get_json()  # Utiliza get_json para parsear el JSON recibido
    vote_type = data['vote']
    scoring(vote_type, data['image_id'], data['color_relevance'])
    print(f"Voto registrado: {vote_type}")  # Imprime el voto para propósitos de debug
    return jsonify({"status": "success", "vote": vote_type})

@app.route('/similarity', methods=['POST'])
def similarity_req():
    data = request.get_json()
    image_id = int(data['image_id'])
    color_value = int(data['color_relevance'])
    image_embeddings_colored = np.concatenate([image_embeddings, color_embeddings*color_value], axis=1)
    index = faiss.IndexFlatL2(image_embeddings_colored.shape[1])
    index.add(image_embeddings_colored)
    D, I = index.search(image_embeddings_colored[image_id:image_id+1], 5)
    similar_images = [{"id": int(idx), "url": images['path_to_image'][idx]} for idx in I[0]]
    return jsonify(similar_images)

if __name__ == "__main__":
    app.run(debug=True)
