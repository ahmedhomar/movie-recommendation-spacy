import spacy


# Load the installed model "en_core_web_md"
nlp = spacy.load("en_core_web_md")

# Movie description for comparison in the model.
planet_hulk_desc = """Will he save their world or destroy it? When the Hulk
becomes too dangerous for the Earth,the Illuminati trick Hulk into a shuttle
and launch him into space to a planet where the Hulk can live in peace.
Unfortunately,Hulk lands on the planet Sakaar where he is sold into slavery
and trained as a gladiator."""

# Create doc object
model_desc = nlp(planet_hulk_desc)

# Function:
def get_watch_next(movie_desc):
    """Returns a movie recommendation based on a movie description"""

    # Read the movies from movies.txt and store in a list.
    with open("movies.txt", "r", encoding="utf-8") as movie_file:
        movies = movie_file.readlines()
        movies = [movie.strip() for movie in movies]

    # Dict: key = movie, value = similarity (0-1).
    similarity_dict = {}

    # Run the model on each movie.
    for movie in movies:
        similarity = nlp(movie).similarity(movie_desc)
        similarity_dict[movie] = similarity

    # Find movie with highest similarity to Planet Hulk:
    highest_similarity = max(similarity_dict.values())

    # Print the movie recommendation
    for key, value in similarity_dict.items():
        if value == highest_similarity:
            print(
                f"""
You should really watch:

{key}
"""
            )


get_watch_next(model_desc)
