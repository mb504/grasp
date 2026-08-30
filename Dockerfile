FROM python:3.12-slim

WORKDIR /grasp
ENV PYTHONUNBUFFERED=1 \
  GRASP_INDEX_DIR=/opt/grasp

# Copy files
COPY . .

# Install GRASP
RUN pip install --no-cache-dir .

# Run GRASP by default; override flags via `docker run grasp -- <args>`
ENTRYPOINT ["grasp"]
CMD ["--help"]


# to build the container: 'docker build -t grasp .'

# to run: 'docker run grasp'

# Below is an example to run grasp entity-linking with an example input.
# --log-level is set to DEBUG to view the linking process
# you need to set the correct url of the LLM API in configs/run.yaml

# to try out different annotation methods, go to configs/run.yaml and change the method in:
# task_kwargs:
#   entity-linking:
#     method: matching
#     know_before_annotate: True
# to one of: matching, indices, markdown, prefix.
# know_before_annotate means the entity needs to be found in the knowledge graph before it can be used for annotation

# In the input the following fields can be used:
# data: the text that should be linked as a string
# annotate_from: from which character on the text should be annotated
# annotate_up_to: up to which character the text should be annotated
# special_instructions: choose e.g. what entities should be linked like: link only historical events

# docker run grasp --log-level DEBUG run configs/run.yaml --task entity-linking --input-format json --input '{"text": {"data": "SMS Schwaben was the fourth of five ships of the Wittelsbach class of pre-dreadnought battleships of the Imperial German Navy. Built at the Imperial Dockyard in Wilhelmshaven, she was laid down in 1900 and completed in April 1904. Possessing a main battery of four 24-centimeter (9.4 in) guns, the ship had a top speed of 18 knots (33 km/h; 21 mph). Schwaben spent her early career as a gunnery training ship or participating in large-scale fleet exercises. At the start of World War I in August 1914, the sister ships were mobilized as IV Battle Squadron. Schwaben served in the North Sea and then the Baltic Sea until the threat from British submarines forced her to withdraw in 1916. For the remainder of the war, she was an engineering training ship for navy cadets. After the war, she was retained as a depot ship for F-type minesweepers in the Baltic from 1919 until June 1920, but was stricken from the navy list in March 1921 and sold for scrap.", "annotate_from": 30, "annotate_up_to": 300, "special_instructions": null}}'
