import streamlit as st
import os
import time

from io import BytesIO
from base64 import b64encode, b64decode
from PIL import Image
from octoai.client import Client
from streamlit_image_select import image_select

# FIXME: comment back in when Langchain is updated
# from langchain.llms.octoai_endpoint import OctoAIEndpoint
from octoai_endpoint import OctoAIEndpoint
from langchain.llms import OpenAI
from langchain.llms.fireworks import Fireworks
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator


llm_map = {
    # "OctoML-CodeLlama-34B-instruct": "codellama-34b-instruct-fp16",
    "OctoML-CodeLlama-34B-instruct-int4": "codellama-34b-instruct-int4",
    "OctoML-CodeLlama-13B-instruct": "codellama-13b-instruct-fp16",
    "OctoML-CodeLlama-7B-instruct": "codellama-7b-instruct-fp16",
    "OctoML-Mistral-7B-instruct": "mistral-7b-instruct-fp16",
    "OctoML-Llama2-13B-chat": "llama-2-13b-chat-fp16",
    # "OctoML-Llama2-70B-chat": "llama-2-70b-chat-fp16",
    # "OctoML-Llama2-70B-chat-int4": "llama-2-70b-chat-int4",
    "GPT4-turbo": "gpt-4-1106-preview",
    "GPT3.5-turbo": "gpt-3.5-turbo-1106",
    "Fireworks-lllama-v2-7b-chat": "accounts/fireworks/models/llama-v2-7b-chat",
    "Fireworks-llama-v2-13b-chat": "accounts/fireworks/models/llama-v2-13b-chat",
    "Fireworks-llama-v2-70b-chat": "accounts/fireworks/models/llama-v2-70b-chat",
    "Fireworks-llama-v2-34b-code": "accounts/fireworks/models/llama-v2-34b-code",
    "Fireworks-llama-v2-34b-code-instruct": "accounts/fireworks/models/llama-v2-34b-code-instruct",
    "Fireworks-mistral-7b": "accounts/fireworks/models/mistral-7b",
    "Fireworks-mistral-7b-instruct-4k": "accounts/fireworks/models/mistral-7b-instruct-4k",
}

class CharacterClasses(BaseModel):
    class_1: str = Field(description="character class 1")
    class_2: str = Field(description="character class 2")
    class_3: str = Field(description="character class 3")
    description_1: str = Field(description="description for class 1")
    description_2: str = Field(description="description for class 2")
    description_3: str = Field(description="description for class 3")

class GuideResponse(BaseModel):
    setup: str = Field(description="narrator describing the situation")
    option_a: str = Field(description="choice a")
    option_b: str = Field(description="choice b")
    option_c: str = Field(description="choice c")
    decision: str = Field(description="narrator asking the player to make a choice", default="Enter your choice by typing A, B, or C.")

class AdventureParams(BaseModel):
    goal: str = Field(description="The goal of the turn by turn adventure")
    setting: str = Field(description="The setting")

SDXL_payload = {
    "prompt": "",
    "negative_prompt": "",
    "style_preset": "cinematic",
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "sampler": "DPM_PLUS_PLUS_SDE_KARRAS",
    "use_refiner": True,
    "steps": 20,
    "cfg_scale": 7.5
}

def image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = b64encode(buffered.getvalue()).decode("utf-8")
    return img_b64

def generate_an_adventure(topic, llm):
     # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=AdventureParams)
    prompt = PromptTemplate(
        template="Answer the user query.\n{query}\n{format_instructions}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    attempts = 0
    responses = []
    while attempts < 5:
        try:
            response = conversation.run(
                "Given the following description of a drawing, come up with a fun turn by turn adventure game concept by very concisely describing the goal and setting of the adventure for a child: \n Description: {}. \n Answer: \n".format(topic))
            print(response)
            responses.append(responses)
            # Test that it is properly formatted
            adventure_params = parser.parse(response)
            break
        except:
            attempts += 1

    if attempts==5:
        st.text_area("All attempts to get a JSON formatted output out of our LLM have failed. Showing the output of the LLM invocations for debugging:", value=responses)
        adventure_params = None

    return adventure_params

def generate_character_classes(goal, setting, llm):
     # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=CharacterClasses)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    attempts = 0
    responses = []
    while attempts < 5:
        try:
            response = conversation.run("Describe 3 character classes for a turn by turn game, set in {} where the objective is to {}.".format(setting, goal))
            print(response)
            responses.append(responses)
            # Test that it is properly formatted
            character_class_info = parser.parse(response)
            break
        except:
            attempts += 1

    if attempts==5:
        st.text_area("All attempts to get a JSON formatted output out of our LLM have failed. Showing the output of the LLM invocations for debugging:", value=responses)
        character_class_info = None

    return character_class_info

def generate_character_images(session_state, progress_bar, percentage, update):
    # Derive inputs
    if 'img_file_buffer' in session_state:
        user_img = Image.open(session_state['img_file_buffer'])
    else:
        user_img = None
    class_info = session_state["class_info"]
    setting = st.session_state['setting']

    # generate images
    sdxl_futures = []
    fs_futures = []

    # Submit SDXL queries
    SDXL_payload["style_preset"] = session_state['style_preset']
    SDXL_payload["prompt"] = "((portrait of {})), {}, {}".format(class_info.class_1, class_info.description_1, setting)
    future = prod_client.infer_async(endpoint_url="https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
    sdxl_futures.append(future)
    SDXL_payload["prompt"] = "((portrait of {})), {}, {}".format(class_info.class_2, class_info.description_2, setting)
    future = prod_client.infer_async(endpoint_url="https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
    sdxl_futures.append(future)
    SDXL_payload["prompt"] = "((portrait of {})), {}, {}".format(class_info.class_3, class_info.description_3, setting)
    future = prod_client.infer_async(endpoint_url="https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
    sdxl_futures.append(future)

    percentage += 10
    progress_bar.progress(percentage, update)

    # Get SDXL images
    images = []
    for sdxl_future in sdxl_futures:
        while not prod_client.is_future_ready(sdxl_future):
            time.sleep(0.1)
        result = prod_client.get_future_result(sdxl_future)
        sdxl_str = result["images"][0]["image_b64"]
        if user_img:
            fs_future = prod_client.infer_async(
                endpoint_url="https://octoshop-faceswap-4jkxk521l3v1.octoai.cloud/predict",
                inputs={
                    "src_image": image_to_base64(user_img),
                    "dst_image": sdxl_str
                }
            )
            fs_futures.append(fs_future)
        else:
            image = Image.open(BytesIO(b64decode(sdxl_str)))
            images.append(image)
        percentage += 10
        progress_bar.progress(percentage, update)

    # Get FS images
    if user_img:
        for fs_future in fs_futures:
            while not prod_client.is_future_ready(fs_future):
                time.sleep(0.1)
            result = prod_client.get_future_result(fs_future)
            image_str = result["completion"]["image"]
            image = Image.open(BytesIO(b64decode(image_str)))
            images.append(image)
            percentage += 10
            progress_bar.progress(percentage, update)

    return images

def init_llm(session_state, JSON=False):
    if JSON:
        json_addendum = "The response should be formatted as a JSON instance."
    else:
        json_addendum = ""
    # Init LLM
    if "Fireworks" in session_state['llm_select']:
        llm = Fireworks(
            fireworks_api_key=os.environ["FIREWORKS_API_KEY"],
            model=llm_map[session_state['llm_select']],
            max_tokens=512
        )
    elif "GPT" in session_state['llm_select']:
        llm = OpenAI(
            model_name=llm_map[session_state['llm_select']],
            temperature=0.0,
            openai_api_key=os.environ["OPENAI_API_KEY"]
        )
    else:
        llm = OctoAIEndpoint(
            octoai_api_token=os.environ["OCTOAI_API_TOKEN"],
            endpoint_url="https://text.octoai.run/v1/chat/completions",
            model_kwargs={
                "model": llm_map[session_state['llm_select']],
                "messages": [
                    {
                        "role": "system",
                        "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. {}".format(json_addendum),
                    }
                ],
                "stream": False,
                "max_tokens": 512,
                "presence_penalty": 0,
                "temperature": 0.1,
                "top_p": 0.9
            },
        )
    return llm

# Init OctoAI client
prod_client = Client(os.environ["OCTOAI_API_TOKEN"])

# Streamlit page layout
if "page" not in st.session_state:
    st.session_state.page = 0

def nextpage(): st.session_state.page += 1

def nextpage_3(): st.session_state.page = 3

def restart(): st.session_state.page = 0

placeholder = st.empty()

if st.session_state.page == 0:
    st.session_state['llm_select'] = st.selectbox(
        'Choose the LLM technology to power your adventure',
        list(llm_map.keys()))
    # Init LLM
    st.session_state['llm'] = init_llm(st.session_state, JSON=True)

    st.button("Setup your adventure", on_click=nextpage)

elif st.session_state.page == 1:
    # Slider to determine game length
    st.session_state["turn"] = st.slider(
        "Select how many turns to play until you reach the end of this adventure",
        1, 10, 3)
    # Get user Goals and Setting
    img_sel = image_select(
        label="Select an adventure",
        images=[
            Image.open("assets/llama_gem.png"),
            Image.open("assets/suriving_the_zombie_apocalypse.png"),
            Image.open("assets/win_the_fifa_worldcup.png"),
            Image.open("assets/reach_a_successful_ipo.png"),
            Image.open("assets/win_3_michelin_stars.png"),
            Image.open("assets/land_on_mars.png"),
            Image.open("assets/select_your_own.png"),
            Image.open("assets/create_from_drawing.png"),
        ],
        captions=[
            "Live an epic fantasy adventure",
            "Survive the zombie apocalypse",
            "Win the soccer world cup",
            "Lead your startup to a successful IPO",
            "Earn your restaurant 3 Michelin stars",
            "Land on planet Mars",
            "Build your own adventure!",
            "Derive from a drawing",
        ],
        return_value = "index"
    )
    if img_sel == 7:
        drawing = st.file_uploader("Upload a photo of a drawing", type=["png", "jpg", "jpeg"])
        if drawing:
            drawing_img = Image.open(drawing)
            inputs = {
                "input": {
                    "image": image_to_base64(drawing_img),
                    "mode": "fast"
                }
            }
            response = prod_client.infer(endpoint_url="{}/infer".format("https://clip-interrogator-15-4jkxk521l3v1.octoai.run"), inputs=inputs)
            caption = response["output"]["description"]
            print("CLIP output: {}".format(caption))
            caption = caption.split(",")[0]
            print("Shortened CLIP output: {}".format(caption))

            adventure_params = generate_an_adventure(caption, st.session_state['llm'])
            if adventure_params:
                goal, setting = adventure_params.goal, adventure_params.setting

                # Make the goal and setting editable
                st.write("I've derived some ideas for a turn by turn game from this drawing!")
                st.image(drawing_img)
                st.session_state['goal'] = st.text_input("Describe the ultimate goal of your adventure", value=goal)
                st.session_state['setting'] = st.text_input("Describe the setting of your adventure", value=setting)
                st.session_state['style_preset'] = 'comic-book'

                st.button("Set your character", on_click=nextpage_3)
            else:
                st.button("Restart adventure", on_click=restart)
    else:
        if img_sel == 0:
            st.session_state['goal'] = "find the gem of infinite context window"
            st.session_state['setting'] = "the magical forest of ancient llamas"
            st.session_state['style_preset'] = 'fantasy-art'
        elif img_sel == 1:
            st.session_state['goal'] = "find a cure for the zombie apocalypse"
            st.session_state['setting'] = "the start of a zombie apocalypse in San Franscisco"
            st.session_state['style_preset'] = 'digital-art'
        elif img_sel == 2:
            st.session_state['goal'] = "win the FIFA world cup"
            st.session_state['setting'] = "North America in 2026"
            st.session_state['style_preset'] = 'photographic'
        elif img_sel == 3:
            st.session_state['goal'] = "successfully lead your startup to a successful IPO"
            st.session_state['setting'] = "Seattle in 2019"
            st.session_state['style_preset'] = 'cinematic'
        elif img_sel == 4:
            st.session_state['goal'] = "lead your restaurant to success by attaining 3 michelin stars"
            st.session_state['setting'] = "New York in 2005"
            st.session_state['style_preset'] = 'cinematic'
        elif img_sel == 5:
            st.session_state['goal'] = "be the first human to land on Mars"
            st.session_state['setting'] = "spaceship to mars, 2050"
            st.session_state['style_preset'] = '3d-model'
        elif img_sel == 6:
            st.session_state['goal'] = st.text_input("Describe the ultimate goal of your adventure", value="find the gem of infinite context window")
            st.session_state['setting'] = st.text_input("Describe the setting of your adventure", value="the magical forest of ancient llamas")
            st.session_state['style_preset'] = 'cinematic'

        # Override the style preset
        st.session_state['style_preset'] = 'cinematic'
        st.button("Set your character", on_click=nextpage_3)

elif st.session_state.page == 2:
    # Take a photo!
    img_file_buffer = st.camera_input("Take a picture of yourself to create your character!")

    if img_file_buffer:
        st.session_state['img_file_buffer'] = img_file_buffer
        st.button("Generate your personalized character", on_click=nextpage)
    else:
        st.button("Skip the character personalization", on_click=nextpage)

elif st.session_state.page == 3:

    if "images" not in st.session_state:
        # Display progress bar
        percentage = 10
        update = "Character creation in progress..."
        progress_bar = st.progress(percentage, text=update)

        # Invoke LLM to generate character classes
        st.session_state["class_info"] = generate_character_classes(st.session_state['goal'], st.session_state['setting'], st.session_state['llm'])
        percentage = 30
        update = "Created classes, now generating avatars..."
        progress_bar.progress(percentage, text=update)

        # Generate character images
        st.session_state["images"] = generate_character_images(st.session_state, progress_bar, percentage, update)

        # Empty progress bar and go to next page
        progress_bar.empty()

    if st.session_state["class_info"]:
        st.markdown("## Select your character")
        # Display images
        col0, col1, col2 = st.columns(3)
        col0.image(st.session_state["images"][0])
        col0.markdown('''
            #### {}
            {}
        '''.format(st.session_state["class_info"].class_1, st.session_state["class_info"].description_1))
        col1.image(st.session_state["images"][1])
        col1.markdown('''
            #### {}
            {}
        '''.format(st.session_state["class_info"].class_2, st.session_state["class_info"].description_2))
        col2.image(st.session_state["images"][2])
        col2.markdown('''
            #### {}
            {}
        '''.format(st.session_state["class_info"].class_3, st.session_state["class_info"].description_3))

        # Create a selector
        character_class = st.radio("Select your character class", [st.session_state["class_info"].class_1, st.session_state["class_info"].class_2, st.session_state["class_info"].class_3], index=0, horizontal=True)
        st.session_state["character_name"] = st.text_input("Name your character", value="Ollie")

        st.session_state["character_class"] = character_class
        if character_class == st.session_state["class_info"].class_1:
            st.session_state["character_description"] = st.session_state["class_info"].description_1
            st.session_state["avatar"] = st.session_state["images"][0]
        elif character_class == st.session_state["class_info"].class_2:
            st.session_state["character_description"] = st.session_state["class_info"].description_2
            st.session_state["avatar"] = st.session_state["images"][1]
        elif character_class == st.session_state["class_info"].class_3:
            st.session_state["character_description"] = st.session_state["class_info"].description_3
            st.session_state["avatar"] = st.session_state["images"][2]

        st.button("Start adventure", on_click=nextpage)
    else:
        st.button("Restart adventure", on_click=restart)

elif st.session_state.page == 4:
    if "conversation" not in st.session_state:
        # Enable memory
        # Optionally, specify your own session_state key for storing messages
        st.session_state["msgs"] = StreamlitChatMessageHistory(key="chat_messages")
        st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history")

        # Langchain template
        template = """
You are now the guide of a journey set in {}.
You are guiding a player, a {} named {} (they/them/their) who is seeking to {}.
{}
You must navigate them through challenges, choices, and consequences,
dynamically adapting the tale based on the traveler's decisions.
Your goal is to create a branching narrative experience where each choice
leads to a new path, ultimately determining the player's fate.
""".format(
            st.session_state["setting"],
            st.session_state["character_class"],
            st.session_state["character_name"],
            st.session_state["goal"],
            st.session_state["character_description"]
        )

        template += """
Here are some rules to follow:
1. Always present 3 choices: 1, 2, 3 as a numbered list, each choice is just one sentence long.
2. Have a few paths that lead to success, have some paths that lead to failure.
3. Upon receiving "Conclusion." from the human, generate a response that ends the story indicating to the player their failure or success. Add "Success!" if the player won. Add "Game Over!" if the player failed. I will look for this text to end the game.

Here is the chat history, use this to understand what to say next: {chat_history}

Human: {human_input}
AI:"""

        # Set up a parser + inject instructions into the prompt template.
        # parser = PydanticOutputParser(pydantic_object=GuideResponse)

        # Init prompt template
        st.session_state["prompt"] = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=template,
            # partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        # Init LLM
        st.session_state["llm"] = init_llm(st.session_state, JSON=False)
        st.session_state["conversation"] = LLMChain(
            llm=st.session_state["llm"],
            prompt=st.session_state["prompt"],
            verbose=True,
            memory=st.session_state["memory"]
        )

    if len(st.session_state["msgs"].messages) == 0:
        response = st.session_state["conversation"].run("start")
        st.session_state["msgs"].add_ai_message(response)

    for msg in st.session_state["msgs"].messages:
        if msg.type == "human":
            st.chat_message(msg.type, avatar=st.session_state["avatar"]).write(msg.content)
        else:
            st.chat_message(msg.type).write(msg.content)

    # st.write("Type in \"start\" to start your adventure")
    if user_prompt := st.chat_input():
        st.chat_message("human", avatar=st.session_state["avatar"]).write(user_prompt)
        st.session_state["msgs"].add_user_message(user_prompt)
        st.session_state["turn"] -= 1
        if st.session_state["turn"] == 0:
            response = st.session_state["conversation"].run("{}. Conclusion.".format(user_prompt))
        else:
            response = st.session_state["conversation"].run(user_prompt)
        print(response)

        if "Success!" in response or "Game Over!" in response:
            SDXL_payload["style_preset"] = st.session_state["style_preset"]
            if "Success!" in response:
                SDXL_payload["prompt"] = "((portrait of a {} smiling, celebrating)) {}, {}".format(
                    st.session_state["character_class"],
                    st.session_state["goal"],
                    st.session_state["character_description"],
                )
            elif "Game Over!" in response:
                # Take the last sentence from the response
                SDXL_payload["prompt"] = "((portrait of a {} looking really sad and defeated)) {}, {}".format(
                    st.session_state["character_class"],
                    st.session_state["setting"],
                    st.session_state["character_description"],
                )
            future = prod_client.infer_async(endpoint_url="https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
            while not prod_client.is_future_ready(future):
                time.sleep(0.1)
            result = prod_client.get_future_result(future)
            image_str = result["images"][0]["image_b64"]

            if 'img_file_buffer' in st.session_state:
                user_img = Image.open(st.session_state['img_file_buffer'])
                fs_future = prod_client.infer_async(
                    endpoint_url="https://octoshop-faceswap-4jkxk521l3v1.octoai.cloud/predict",
                    inputs={
                        "src_image": image_to_base64(user_img),
                        "dst_image": image_str
                    }
                )
                while not prod_client.is_future_ready(fs_future):
                    time.sleep(0.1)
                result = prod_client.get_future_result(fs_future)
                image_str = result["completion"]["image"]

            image = Image.open(BytesIO(b64decode(image_str)))
            ai_chat = st.chat_message("ai")
            ai_chat.image(image)
            ai_chat.write(response)
            st.session_state["msgs"].add_ai_message(response)

            st.button("Restart!",on_click=restart)
        else:
            st.chat_message("ai").write(response)
            st.session_state["msgs"].add_ai_message(response)

        # Test that it is properly formatted
        # guide_response.parse(response)
        # ai_reply = st.chat_message("ai")
        # json_response = json.loads(response)
        # ai_reply.write(guide_response.setup)
        # col0, col1, col2 = ai_reply.columns(3)

        # # generate image
        # sdxl_futures = []
        # SDXL_payload["prompt"] = guide_response.option_a
        # future = prod_client.infer_async(endpoint_url="https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
        # sdxl_futures.append(future)
        # SDXL_payload["prompt"] = guide_response.option_b
        # future = prod_client.infer_async(endpoint_url="https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
        # sdxl_futures.append(future)
        # SDXL_payload["prompt"] = guide_response.option_c
        # future = prod_client.infer_async(endpoint_url="https://image.octoai.run/generate/sdxl", inputs=SDXL_payload)
        # sdxl_futures.append(future)

        # # Get images
        # images = []
        # for future in sdxl_futures:
        #     while not prod_client.is_future_ready(future):
        #         time.sleep(0.1)
        #     result = prod_client.get_future_result(future)
        #     image_str = result["images"][0]["image_b64"]
        #     image = Image.open(BytesIO(b64decode(image_str)))
        #     images.append(image)
        # col0.image(images[0])
        # col0.write(guide_response.option_a)
        # col1.image(images[1])
        # col1.write(guide_response.option_b)
        # col2.image(images[2])
        # col2.write(guide_response.option_c)

        # ai_reply.write(guide_response.decision)

# else:
#     with placeholder:
#         st.write("This is the end")
#         st.button("Restart",on_click=restart)