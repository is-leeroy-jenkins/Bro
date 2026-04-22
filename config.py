'''
  ******************************************************************************************
      Assembly:                Bro
      Filename:                config.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="config.py" company="Terry D. Eppler">

	     config.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    config.py
  </summary>
  ******************************************************************************************
'''
import os
import re
import multiprocessing

# ---------- DEFINITIONS -------------------

BRO_LLM_PATH = os.getenv( 'BRO_LLM_PATH' )
BLUE_DIVIDER = "<div style='height:2px;align:left;background:#0078FC;margin:6px 0 10px 0;'></div>"
APP_TITLE = 'Bro'
APP_SUBTITLE = 'Local AI based on Gemma 3'
BASE_DIR = os.path.dirname( os.path.abspath( __file__ ) )
DB_PATH = 'stores/sqlite/bro.db'
MODEL_PATH = r'C:\Users\terry\source\llm\bro\bro-3-1b-it-Q4_K_M.gguf'
DEFAULT_CTX = 4096
CORES = multiprocessing.cpu_count( )
FAVICON = r'resources/images/favicon.ico'
LOGO = r'resources/images/bro_logo.png'
XML_BLOCK_PATTERN = re.compile( r"<(?P<tag>[a-zA-Z0-9_:-]+)>(?P<body>.*?)</\1>", re.DOTALL )
MARKDOWN_HEADING_PATTERN = re.compile( r"^##\s+(?P<title>.+?)\s*$" )
MODES = [ 'Text Generation', 'Document Q&A', 'Semantic Search',
          'Prompt Engineering', 'Data Management' ]

# ---------- DEFINITIONS -------------------

SYSTEM_INSTRUCTIONS = r'''Optional. Gives the model high-level instructions on how it should behave while
		generating a response, including tone, goals, and examples of correct responses. Any
		instructions provided this way will take priority over a prompt in the input parameter.'''

TEMPERATURE = r'''Optional. A number between 0 and 2. Higher values like 0.8 will make the output
		more random, while lower values like 0.2 will make it more focused and deterministic'''

TOP_P = r'''Optional. The maximum cumulative probability of tokens to consider when sampling.
		The model uses combined Top-k and Top-p (nucleus) sampling. Tokens are sorted based on
		their assigned probabilities so that only the most likely tokens are considered.
		Top-k sampling directly limits the maximum number of tokens to consider,
		while Nucleus sampling limits the number of tokens based on the cumulative probability.'''

TOP_K = r'''Optional. The maximum number of tokens to consider when sampling. Gemini models use
		Top-p (nucleus) sampling or a combination of Top-k and nucleus sampling. Top-k sampling considers
		the set of topK most probable tokens. Models running with nucleus sampling don't allow topK setting.
		Note: The default value varies by Model and is specified by theModel.top_p attribute returned
		from the getModel function. An empty topK attribute indicates that the model doesn't apply
		top-k sampling and doesn't allow setting topK on requests.'''

PRESENCE_PENALTY = r'''Optional. Presence penalty applied to the next token's logprobs
		if the token has already been seen in the response. This penalty is binary on/off
		and not dependant on the number of times the token is used (after the first).'''

FREQUENCY_PENALTY = r'''Optional. Frequency penalty applied to the next token's logprobs,
		multiplied by the number of times each token has been seen in the respponse so far.
		A positive penalty will discourage the use of tokens that have already been used,
		proportional to the number of times the token has been used: The more a token is used,
		the more difficult it is for the model to use that token again increasing
		the vocabulary of responses.'''

REPEAT_PENALTY = '''Penalizes repeated tokens to reduce looping and redundant responses.'''

CONTEXT_WINDOW = '''The context window is the maximum length of input a large language model (LLM)
		can consider at once. In the development and maturation of LLM technology expanding the
		context window has been a major goal. The length of a context window is
		measured in tokens. '''

REPEAT_WINDOW = '''"Prompt repetition" is a technique where repeating the entire input prompt
		(e.g., <prompt><prompt>) improves LLM performance on non-reasoning tasks by 21-97%.
		This method allows models to better focus their attention, particularly when references
		at the end of a prompt need to connect back to information at the beginning'''

CPU_CORES = '''Number of CPU threads used for inference; higher values improve speed but increase CPU
            "usage." '''

MAX_TOKENS = '''Maximum number of tokens generated per response.'''

SEED = '''Set to a fixed value for reproducible outputs; use -1 for a random seed each run.'''

PROMPT_ENGINEERING = r'''Prompt engineering is the process of writing effective instructions
		for a model, such that it consistently generates content that meets your requirements.
		Because the content generated from a model is non-deterministic, prompting to get your
		desired output is a mix of art and science. However, you can apply techniques and
		best practices to get good results consistently.
		'''

TEXT_GENERATION = r'''Use a large language model to produce coherent, context-aware natural language
		output in response to user prompts, system instructions, or retrieved document context.
		When a user submits a request—whether it is a general inquiry, a structured analytical task,
		or a document-grounded question—Bro constructs a prompt that may include system directives,
		conversation history, and optionally retrieved content from its vector store. The underlying
		model then generates text according to configurable parameters such as temperature,
		maximum tokens, and response format. This capability enables Bro to function as
		a conversational assistant, analytical explainer, summarizer, drafting tool, and reasoning engine,
		producing structured or narrative outputs tailored to the user’s workflow. '''

DATA_MANAGEMENT = r'''Structured handling, organization, processing of
		user-provided data in a self-contained SQLite Database. It allows uploading of files, extracting and
		normalizing their content, chunking text for semantic processing, generating embeddings,
		storing metadata, and enabling controlled retrieval for downstream features such as Document Q&A
		and Data Analysis. Beyond ingestion, it includes version awareness, indexing, schema inspection
		(where applicable), and the ability to manage or remove stored assets safely. Document
		Management provides the foundational infrastructure that transforms raw files into structured,
		searchable, and model-ready assets, ensuring that Bro’s intelligence features operate
		on reliable, well-governed data rather than unmanaged documents.  '''

RETRIEVAL_AUGMENTATION = '''Retrieval-Augmented Generation (RAG) improves LLM accuracy and relevance
		by fetching up-to-date, external data—such as documents, databases, or web results—and feeding
		it into the prompt before generating a response. It reduces hallucinations and eliminates the
		need to retrain models for new information.'''

SEMANTIC_SEARCH = '''LLM semantic search uses Large Language Models and embedding vectors to retrieve
		information based on conceptual meaning and user intent, rather than strict keyword matching.
		By converting documents and queries into numerical vector embeddings stored in a database,
		systems can find contextually relevant information, enabling more accurate, conversational,
		and nuanced search experiences, often used in RAG (Retrieval-Augmented Generation) systems.'''
