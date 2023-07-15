# This script contains iNat identificator class that takes care of identifying insects using the iNaturalist model

import requests
import utils
import json

class InatObservation():

    def __init__(self, access_token, payload_data: dict):

        # The payload data should be a dictionary of keys and values. There are two options.
        #
        # To use the v1 iNat API you only need to supply dict in this format:
        # {
        #                 "species_guess": species_guess,
        #                 "taxon_id": taxon_id,
        #                 "description": description
        #             }
        # To include additional details in the newly created observation, you can supply additional keys:
        # {
        #                  "species_guess": "Northern Cardinal",
        #                  "taxon_id": 9083,
        #                  "observed_on_string": "2013-01-03",
        #                  "time_zone": "Eastern Time (US & Canada)",
        #                  "description": "what a cardinal",
        #                  "tag_list": "foo,bar",
        #                  "place_guess": "clinton, ct",
        #                  "latitude": 41.27872259999999,
        #                  "longitude": -72.5276073,
        #                  "map_scale": 11,
        #                  "location_is_exact": False,
        #                  "positional_accuracy": 7798,
        #                  "geoprivacy": "obscured",
        #                  "observation_field_values_attributes": [
        #                      {
        #                          "observation_field_id": 5,
        #                          "value": "male"
        #                      }
        #                  ]
        #               }


        # Define logger
        self.logger = utils.log_define()

        self.access_token = access_token

        # Set the API endpoint for creating an observation
        self.photo_url = "https://api.inaturalist.org/v1/observation_photos"

        extra_keys = set(payload_data.keys()) - {"species_guess", "description", "taxon"}
        if extra_keys:
            self.create_observation_legacy(payload_data)
        else:
            self.create_observation_v1(payload_data)

    def create_observation_v1(self, payload_data):

        self.create_url = "https://api.inaturalist.org/v1/observations"

        # Set the headers for the request
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        # Set the payload data for the observation
        payload = {
            "observation": payload_data
        }

        # Send the POST request to create the observation
        response = requests.post(self.create_url, data=json.dumps(payload), headers=headers)

        # Check the response status code
        if response.status_code == 200:
            # Successful observation creation
            self.logger.info(response.json())
            self.response_json = response.json()
            self.observation_id = response.json()["id"]
            self.observation_uuid = response.json()["uuid"]
            self.logger.info(
                f"Observation created successfully. Observation ID: {self.observation_id}, Observation uuID: {self.observation_uuid}")
        else:
            # Failed to create the observation
            self.logger.info("Failed to create the observation.")
    def create_observation_legacy(self, payload_data):

        # Set the API endpoint for creating an observation
        self.create_url = "https://www.inaturalist.org/observations.json"

        # Set the payload for the observation
        payload = {
            "observation": payload_data
        }

        # Set the headers with authentication
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }

        # Send the POST request with authentication
        response = requests.post(self.create_url, json=payload, headers=headers)

        # Check the response status code
        if response.status_code == 200:
            # Successful observation creation
            self.logger.info(response.json())
            self.response_json = response.json()[0]
            self.observation_id = response.json()[0]["id"]
            self.observation_uuid = response.json()[0]["uuid"]
            self.logger.info(
                f"Observation created successfully. Observation ID: {self.observation_id}, Observation uuID: {self.observation_uuid}")
        else:
            # Failed to create the observation
            self.logger.info(response.json())
            self.logger.info("Failed to create the observation.")

    def upload_image(self, image_file_path):

        access_token = self.access_token
        observation_id = self.observation_id
        url = self.photo_url

        # Set the headers for the request
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }

        # Set the payload data for the multipart/form-data
        payload = {
            "observation_photo[observation_id]": observation_id
        }

        # Set the files data for the image file
        files = {
            "file": open(image_file_path, "rb")
        }

        # Send the POST request to append the image to the observation
        response = requests.post(url, data=payload, files=files, headers=headers)

        # Check the response status code
        if response.status_code == 200:
            # Successful image append
            observation_photo_id = response.json()["id"]
            self.logger.info(f"Image appended successfully. Observation Photo ID: {observation_photo_id}")
        else:
            # Failed to append the image
            self.logger.info("Failed to append the image.")


class InatIdentificator():

    def __init__(self, client_id, client_secret, username, password):

        # client_id = "LWjPdw733dsLPmr_05T9GWYX-9_qZ40TcGNtUq4ZPvk"
        # client_secret = 'HhXxWAsXVm-5SQHth_TEKZc2JAf4mUkKi9oFS6DuSzM'
        # username = 'petaschlup@seznam.cz'
        # password = '8*u;n2H5-gLQYV]'

        # Define logger
        self.logger = utils.log_define()

        # First log
        self.logger.debug("iNaturalist Identificator created")

        # Define variables
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        token_url = 'https://www.inaturalist.org/oauth/token'

        self.observations = []

        # Get access token
        self.auth_success, self.access_token = self.get_access_token(token_url)

    def get_access_token(self, token_url):

        access_token = None
        success = False

        payload = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'password',
            'username': self.username,
            'password': self.password
        }

        try:
            response = requests.post(token_url, data=payload)
            access_token = response.json()['access_token']
            print(f"Access token retrieved: {access_token}")
            success = True
            return success, access_token
        except KeyError:
            print(f"Access token could not be retrieved: {response.json()['error_description']}")
            return success, access_token

    def create_observation(self, payload_data: dict):

        observation = InatObservation(self.access_token, payload_data)
        self.observations.append(observation)

        return observation

    def identify_insect(self, image_path):

        # Step 1: Upload the image to iNaturalist
        upload_url = "https://www.inaturalist.org/observations.json"
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        files = {
            "observation[photos_attributes][][file]": open(image_path, "rb")
        }
        response = requests.post(upload_url, files=files, headers=headers)

        if response.status_code == 200:
            print(response.json())
            observation_id = response.json()[0]["id"]
            observation_uuid = response.json()[0]["uuid"]
            print(f"Observation created successfully. Observation ID: {observation_id}, Observation uuID: {observation_uuid}")
        else:
            print("Failed to create observation")



        # Step 2: Identify the uploaded observation
        identify_url = f"https://www.inaturalist.org/observations/{observation_id}/identifications.json"
        payload = {
            "identification[body]": "Insect identification request",
            "identification[taxon_id]": 47157
            # Insecta taxon ID, change if you want to limit the search to a specific taxon
        }

        response = requests.post(identify_url, data=payload, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data:
                identification = data[0]
                species_name = identification["taxon"]["name"]
                order_name = identification["taxon"]["preferred_common_name"]
                print(f"Species: {species_name}")
                print(f"Order: {order_name}")
            else:
                print("No identification results found.")
        else:
            print("Failed to run identification.")


# client_id = "LWjPdw733dsLPmr_05T9GWYX-9_qZ40TcGNtUq4ZPvk"
# client_secret = 'HhXxWAsXVm-5SQHth_TEKZc2JAf4mUkKi9oFS6DuSzM'
# username = 'petaschlup@seznam.cz'
# password = '8*u;n2H5-gLQYV]'
# identificator = InatIdentificator(client_id, client_secret, username, password)
#
# observation_data = {
#                          "species_guess": "Amegilla quadrifasciata",
#                          "taxon_id": 47158,
#                          "observed_on_string": "2022-05-24",
#                          "time_zone": "Prague",
#                          "description": "large bee",
#                          "tag_list": "GR2",
#                          "place_guess": "Lesvos, GR",
#                          "latitude": 39.166666,
#                          "longitude": 26.333332,
#                          "map_scale": 11,
#                          "location_is_exact": False,
#                          "positional_accuracy": 1000,
#                          "geoprivacy": "obscured",
#                       }
#
# obs = identificator.create_observation(observation_data)
# obs.upload_image("img.jpg")