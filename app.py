'''
	******************************************************************************************
	    Assembly:                Bro
	    Filename:                app.py
	    Author:                  Terry D. Eppler
	    Created:                 05-31-2024
	
	    Last Modified By:        Terry D. Eppler
	    Last Modified On:        05-01-2025
	******************************************************************************************
	<copyright file="app.py" company="Terry D. Eppler">
	
	           Bro is a data analysis tool integrating various Generative GPT, Text-Processing, and
	           Machine-Learning algorithms for federal analysts.
	           Copyright ©  2023 Terry Eppler
	
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
	  app.py
	</summary>
	******************************************************************************************
'''
from __future__ import annotations

import base64
import hashlib
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import fitz  # pymupdf
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

import config as cfg

# ==============================================================================
# Model Path Resolution
# ==============================================================================
MODEL_PATH_OBJ = Path( cfg.MODEL_PATH )

if not MODEL_PATH_OBJ.exists( ):
    st.error( f'Model not found at {cfg.MODEL_PATH}' )
    st.stop( )
    
# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'mode' not in st.session_state:
	st.session_state[ 'mode' ] = ''

if 'messages' not in st.session_state:
	st.session_state[ 'messages' ] = [ ]

if 'system_instructions' not in st.session_state:
	st.session_state[ 'system_instructions' ] = ''

if 'context_window' not in st.session_state:
	st.session_state[ 'context_window' ] = 0

if 'cpu_threads' not in st.session_state:
	st.session_state[ 'cpu_threads' ] = 0

if 'max_tokens' not in st.session_state:
	st.session_state[ 'max_tokens' ] = 0

if 'temperature' not in st.session_state:
	st.session_state[ 'temperature' ] = 0.0

if 'top_percent' not in st.session_state:
	st.session_state[ 'top_percent' ] = 0.0

if 'top_k' not in st.session_state:
	st.session_state[ 'top_k' ] = 0

if 'frequency_penalty' not in st.session_state:
	st.session_state[ 'frequency_penalty' ] = 0.0

if 'presense_penalty' not in st.session_state:
	st.session_state[ 'presense_penalty' ] = 0.0

if 'repeat_penalty' not in st.session_state:
	st.session_state[ 'repeat_penalty' ] = 0.0

if 'repeat_window' not in st.session_state:
	st.session_state[ 'repeat_window' ] = 0

if 'random_seed' not in st.session_state:
	st.session_state[ 'random_seed' ] = 0

if 'basic_docs' not in st.session_state:
	st.session_state[ 'basic_docs' ] = [ ]

if 'use_semantic' not in st.session_state:
	st.session_state[ 'use_semantic' ] = False

if 'is_grounded' not in st.session_state:
	st.session_state[ 'is_grounded' ] = False

if 'selected_prompt_id' not in st.session_state:
	st.session_state[ 'selected_prompt_id' ] = ''

if 'pending_system_prompt_name' not in st.session_state:
	st.session_state[ 'pending_system_prompt_name' ] = ''
	
#-------- DOCQNA ---------------------

if 'uploaded' not in st.session_state:
	st.session_state[ 'uploaded' ] = [ ]

if 'active_docs' not in st.session_state:
	st.session_state[ 'active_docs' ] = [ ]

if 'doc_bytes' not in st.session_state:
	st.session_state[ 'doc_bytes' ] = { }
	
if 'doc_source' not in st.session_state:
	st.session_state[ 'doc_source' ] = 'uploadlocal'

if 'docqna_vec_ready' not in st.session_state:
	st.session_state[ 'docqna_vec_ready' ] = False

if 'docqna_fingerprint' not in st.session_state:
	st.session_state[ 'docqna_fingerprint' ] = ''

if 'docqna_chunk_count' not in st.session_state:
	st.session_state[ 'docqna_chunk_count' ] = 0
	
if 'docqna_fallback_rows' not in st.session_state:
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	
# ==============================================================================
# UTILITIES
# ==============================================================================
def image_to_base64( path: str ) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def cosine_similarity( a: np.ndarray, b: np.ndarray ) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float( np.dot(a, b) / denom ) if denom else 0.0

def initialize_database( ) -> None:
	"""
		Purpose:
		--------
		Ensure required SQLite tables exist and that the Prompts table contains the
		columns required by the prompt utilities and Prompt Engineering mode.

		Parameters:
		-----------
		None

		Returns:
		--------
		None
	"""
	Path( 'stores/sqlite' ).mkdir( parents=True, exist_ok=True )
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS chat_history
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                role
                TEXT,
                content
                TEXT
            )
			"""
		)
		
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS embeddings
            (
                id
                INTEGER
                PRIMARY
                KEY
                AUTOINCREMENT,
                chunk
                TEXT,
                vector
                BLOB
            )
			"""
		)
		
		conn.execute(
			"""
            CREATE TABLE IF NOT EXISTS Prompts
            (
                PromptsId
                INTEGER
                NOT
                NULL
                PRIMARY
                KEY
                AUTOINCREMENT,
                Caption
                TEXT,
                Name
                TEXT
            (
                80
            ),
                Text TEXT,
                Version TEXT
            (
                80
            ),
                ID TEXT
            (
                80
            )
                )
			"""
		)
		
		prompt_columns = [ row[ 1 ] for row in
		                   conn.execute( 'PRAGMA table_info("Prompts");' ).fetchall( ) ]
		
		if 'Caption' not in prompt_columns:
			conn.execute( 'ALTER TABLE "Prompts" ADD COLUMN "Caption" TEXT;' )
		
		conn.commit( )

# -------- CHAT/TEXT UTILITIES --------------------

def normalize_text( text: str ) -> str:
	"""
		
		Purpose
		-------
		Normalize text by:
			• Converting to lowercase
			• Removing punctuation except sentence delimiters (. ! ?)
			• Ensuring clean sentence boundary spacing
			• Collapsing whitespace
	
		Parameters
		----------
		text: str
	
		Returns
		-------
		str
		
	"""
	if not text:
		return ""
	
	# Lowercase
	text = text.lower( )
	
	# Remove punctuation except . ! ?
	text = re.sub( r"[^\w\s\.\!\?]", "", text )
	
	# Ensure single space after sentence delimiters
	text = re.sub( r"([.!?])\s*", r"\1 ", text )
	
	# Normalize whitespace
	text = re.sub( r"\s+", " ", text ).strip( )
	
	return text

def chunk_text( text: str, size: int = 1200, overlap: int = 200 ) -> List[ str ]:
	chunks, i = [ ], 0
	while i < len( text ):
		chunks.append( text[ i:i + size ] )
		i += size - overlap
	return chunks

def convert_xml( text: str ) -> str:
	"""
		
			Purpose:
			_________
			Convert XML-delimited prompt text into Markdown by treating XML-like
			tags as section delimiters, not as strict XML.
	
			Parameters:
			-----------
			text (str) - Prompt text containing XML-like opening and closing tags.
	
			Returns:
			---------
			Markdown-formatted text using level-2 headings (##).
	"""
	markdown_blocks: List[ str ] = [ ]
	for match in cfg.XML_BLOCK_PATTERN.finditer( text ):
		raw_tag: str = match.group( "tag" )
		body: str = match.group( "body" ).strip( )
		
		# Humanize tag name for Markdown heading
		heading: str = raw_tag.replace( "_", " " ).replace( "-", " " ).title( )
		markdown_blocks.append( f"## {heading}" )
		if body:
			markdown_blocks.append( body )
	return "\n\n".join( markdown_blocks )

def markdown_converter( text: Any ) -> str:
	"""
		Purpose:
		--------
		Convert between Markdown headings and simple XML-like heading tags.
	
		Behavior:
		---------
		Auto-detects direction:
		  - If <h1>...</h1> / <h2>...</h2> ... exist, converts to Markdown (# / ## / ###).
		  - Otherwise converts Markdown headings (# / ## / ###) to <hN>...</hN> tags.
	
		Parameters:
		-----------
		text : Any
			Source text. Non-string values return "".
	
		Returns:
		--------
		str
			Converted text.
	"""
	if not isinstance( text, str ) or not text.strip( ):
		return ""
	
	# Normalize newlines
	src = text.replace( "\r\n", "\n" ).replace( "\r", "\n" )
	
	htag_pattern = re.compile( r"<h([1-6])>(.*?)</h\1>", flags=re.IGNORECASE | re.DOTALL )
	md_heading_pattern = re.compile( r"^(#{1,6})[ \t]+(.+?)[ \t]*$", flags=re.MULTILINE )
	
	# ------------------------------------------------------------------
	# Direction detection
	# ------------------------------------------------------------------
	contains_htags = bool( htag_pattern.search( src ) )
	
	# ------------------------------------------------------------------
	# XML-like heading tags -> Markdown headings
	# ------------------------------------------------------------------
	if contains_htags:
		def _htag_to_md( match: re.Match ) -> str:
			level = int( match.group( 1 ) )
			content = match.group( 2 ).strip( )
			
			# Preserve inner newlines safely by collapsing interior whitespace
			# while keeping content readable.
			content = re.sub( r"[ \t]+\n", "\n", content )
			content = re.sub( r"\n[ \t]+", "\n", content )
			
			return f"{'#' * level} {content}"
		
		out = htag_pattern.sub( _htag_to_md, src )
		return out.strip( )
	
	# ------------------------------------------------------------------
	# Markdown headings -> XML-like heading tags
	# ------------------------------------------------------------------
	def _md_to_htag( match: re.Match ) -> str:
		hashes = match.group( 1 )
		content = match.group( 2 ).strip( )
		level = len( hashes )
		return f"<h{level}>{content}</h{level}>"
	
	out = md_heading_pattern.sub( _md_to_htag, src )
	return out.strip( )

def inject_response_css( ) -> None:
	"""
	
		Purpose:
		_________
		Set the the format via css.
		
	"""
	st.markdown(
		"""
		<style>
		/* Chat message text */
		.stChatMessage p {
			color: rgb(220, 220, 220);
			font-size: 1rem;
			line-height: 1.6;
		}

		/* Headings inside chat responses */
		.stChatMessage h1 {
			color: rgb(0, 120, 252); /* DoD Blue */
			font-size: 1.6rem;
		}

		.stChatMessage h2 {
			color: rgb(0, 120, 252);
			font-size: 1.35rem;
		}

		.stChatMessage h3 {
			color: rgb(0, 120, 252);
			font-size: 1.15rem;
		}
		
		.stChatMessage a {
			color: rgb(0, 120, 252); /* DoD Blue */
			text-decoration: underline;
		}
		
		.stChatMessage a:hover {
			color: rgb(80, 160, 255);
		}

		</style>
		""", unsafe_allow_html=True )

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True, )

def save_message( role: str, content: str ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( 'INSERT INTO chat_history (role, content) VALUES (?, ?)', (role, content) )

def load_history( ) -> List[ Tuple[ str, str ] ]:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		return conn.execute( 'SELECT role, content FROM chat_history ORDER BY id' ).fetchall( )

def clear_history( ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM chat_history" )

#-------- PROMPT ENGINEERING UTILITIES ----------------

def fetch_prompt_names( db_path: str ) -> list[ str ]:
	"""
		Purpose:
		--------
		Retrieve template names from Prompts table.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
	
		Returns:
		--------
		list[str]
			Sorted prompt names.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Caption FROM Prompts ORDER BY PromptsId;" )
		rows = cur.fetchall( )
		conn.close( )
		return [ r[ 0 ] for r in rows if r and r[ 0 ] is not None ]
	except Exception:
		return [ ]

def fetch_prompt_text( db_path: str, name: str ) -> str | None:
	"""
		Purpose:
		--------
		Retrieve template text by name.
	
		Parameters:
		-----------
		db_path : str
			SQLite database path.
		name : str
			Template name.
	
		Returns:
		--------
		str | None
			Prompt text if found.
	"""
	try:
		conn = sqlite3.connect( db_path )
		cur = conn.cursor( )
		cur.execute( "SELECT Text FROM Prompts WHERE Caption = ?;", (name,) )
		row = cur.fetchone( )
		conn.close( )
		return str( row[ 0 ] ) if row and row[ 0 ] is not None else None
	except Exception:
		return None

def fetch_prompts_df( ) -> pd.DataFrame:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		df = pd.read_sql_query(
			"SELECT PromptsId, Caption,  Name, Version, ID FROM Prompts ORDER BY PromptsId DESC",
			conn )
	df.insert( 0, "Selected", False )
	return df

def fetch_prompt_by_id( pid: int ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE PromptsId=?",
			(pid,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def fetch_prompt_by_name( name: str ) -> Dict[ str, Any ] | None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		cur = conn.execute(
			"SELECT PromptsId, Caption, Name, Text, Version, ID FROM Prompts WHERE Caption=?",
			(name,)
		)
		row = cur.fetchone( )
		return dict( zip( [ c[ 0 ] for c in cur.description ], row ) ) if row else None

def insert_prompt( data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			'INSERT INTO Prompts (Caption, Name, Text, Version, ID) VALUES (?, ?, ?, ?, ?)',
			(data[ 'Caption' ], data[ 'Name' ], data[ 'Text' ], data[ 'Version' ], data[ 'ID' ]) )
		
def update_prompt( pid: int, data: Dict[ str, Any ] ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute(
			"UPDATE Prompts SET Caption=?, Name=?, Text=?, Version=?, ID=? WHERE PromptsId=?",
			(data[ "Caption" ], data[ "Name" ], data[ "Text" ], data[ "Version" ], data[ "ID" ],
			 pid)
		)

def delete_prompt( pid: int ) -> None:
	with sqlite3.connect( cfg.DB_PATH ) as conn:
		conn.execute( "DELETE FROM Prompts WHERE PromptsId=?", (pid,) )

def build_prompt( user_input: str ) -> str:
	"""
		Purpose:
		--------
		Build a llama.cpp-compatible prompt using the application's system instructions, optional
		retrieval context (semantic + basic RAG), and the current in-memory chat history.

		Parameters:
		-----------
		user_input : str
			The current user turn to append to the prompt.

		Returns:
		--------
		str
			A fully constructed prompt in chat template format.
	"""
	system_instructions = st.session_state.get( 'system_instructions', '' )
	use_semantic = bool( st.session_state.get( 'use_semantic', False ) )
	basic_docs = st.session_state.get( 'basic_docs', [ ] )
	messages = st.session_state.get( 'messages', [ ] )
	
	top_k_value = int( st.session_state.get( 'top_k', 0 ) )
	if top_k_value <= 0:
		top_k_value = 4
	
	prompt = f"<|system|>\n{system_instructions}\n</s>\n"
	
	if use_semantic:
		with sqlite3.connect( cfg.DB_PATH ) as conn:
			rows = conn.execute( "SELECT chunk, vector FROM embeddings" ).fetchall( )
		
		if rows:
			q = embedder.encode( [ user_input ] )[ 0 ]
			scored = [ (c, cosine_similarity( q, np.frombuffer( v ) )) for c, v in rows ]
			for c, _ in sorted( scored, key=lambda x: x[ 1 ], reverse=True )[ :top_k_value ]:
				prompt += f"<|system|>\n{c}\n</s>\n"
	
	for d in basic_docs[ :6 ]:
		prompt += f"<|system|>\n{d}\n</s>\n"
	
	if isinstance( messages, list ):
		for msg in messages:
			role = ''
			content = ''
			
			if isinstance( msg, tuple ) or isinstance( msg, list ):
				if len( msg ) == 2:
					role = str( msg[ 0 ] or '' ).strip( )
					content = str( msg[ 1 ] or '' )
			elif isinstance( msg, dict ):
				role = str( msg.get( 'role', '' ) or '' ).strip( )
				content = str( msg.get( 'content', '' ) or '' )
			
			if role:
				prompt += f"<|{role}|>\n{content}\n</s>\n"
	
	prompt += f"<|user|>\n{user_input}\n</s>\n<|assistant|>\n"
	return prompt

def run_llm_turn( user_input: str, temperature: float, top_p: float, repeat_penalty: float,
		max_tokens: int, stream: bool, output: Any | None = None ) -> str:
	"""
		Purpose:
		--------
		Run a single LLM turn using the application's shared prompt builder and either stream or
		return the full response text.

		Parameters:
		-----------
		user_input : str
			The user turn (already constructed, including any document/RAG context if applicable).
		temperature : float
			Sampling temperature.
		top_p : float
			Nucleus sampling probability.
		repeat_penalty : float
			Repeat penalty.
		max_tokens : int
			Maximum tokens to generate.
		stream : bool
			When True, stream tokens to the provided Streamlit placeholder.
		output : Any | None
			A Streamlit placeholder (e.g., st.empty()) used for streaming output.

		Returns:
		--------
		str
			The assistant response text.
	"""
	if user_input is None:
		return ''
	
	prompt = build_prompt( user_input )
	if not stream:
		resp = llm(
			prompt,
			stream=False,
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			repeat_penalty=repeat_penalty,
			stop=[ '</s>' ]
		)
		text = (resp.get( 'choices', [ { 'text': '' } ] )[ 0 ].get( 'text', '' ) or '')
		return text.strip( )
	
	buf = ''
	if output is None:
		output = st.empty( )
	
	for chunk in llm(
			prompt,
			stream=True,
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			repeat_penalty=repeat_penalty,
			stop=[ '</s>' ]
	):
		buf += chunk[ 'choices' ][ 0 ][ 'text' ]
		output.markdown( buf + '▌' )
	
	output.markdown( buf )
	return buf.strip( )

# ----------- DATABASE UTILITIES -------------------------

def create_connection( ) -> sqlite3.Connection:
	return sqlite3.connect( cfg.DB_PATH )

def list_tables( ) -> List[ str ]:
	with create_connection( ) as conn:
		_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
		rows = conn.execute( _query ).fetchall( )
		return [ r[ 0 ] for r in rows ]

def create_schema( table: str ) -> List[ Tuple ]:
	with create_connection( ) as conn:
		return conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )

def read_table( table: str, limit: int=None, offset: int=0 ) -> pd.DataFrame:
	query = f'SELECT rowid, * FROM "{table}"'
	if limit:
		query += f" LIMIT {limit} OFFSET {offset}"
	with create_connection( ) as conn:
		return pd.read_sql_query( query, conn )

def drop_table( table: str ) -> None:
	"""
		Purpose:
		--------
		Safely drop a table if it exists.
	
		Parameters:
		-----------
		table : str
			Table name.
	"""
	if not table:
		return
	
	with create_connection( ) as conn:
		conn.execute( f'DROP TABLE IF EXISTS "{table}";' )
		conn.commit( )

def rename_table( old_name: str, new_name: str ) -> None:
	"""
		Purpose:
		--------
		Rename an existing SQLite table. Attempts native ALTER TABLE rename first; if it fails,
		falls back to a schema-safe rebuild using the original CREATE TABLE statement and
		preserves indexes.

		Parameters:
		-----------
		old_name : str
			Existing table name.

		new_name : str
			New table name.

		Returns:
		--------
		None
	"""
	if not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute( f'ALTER TABLE "{old_name}" RENAME TO "{new_name}";' )
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(old_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(old_name,)
		).fetchall( )
		
		open_paren = create_sql.find( "(" )
		if open_paren == -1:
			raise ValueError( "Malformed CREATE TABLE statement." )
		
		temp_name = f"{new_name}__rebuild_temp"
		
		conn.execute( "BEGIN" )
		conn.execute( f'CREATE TABLE "{temp_name}" {create_sql[ open_paren: ]}' )
		
		cols = [ r[ 1 ] for r in conn.execute( f'PRAGMA table_info("{old_name}");' ).fetchall( ) ]
		col_list = ", ".join( [ f'"{c}"' for c in cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_name}" ({col_list}) SELECT {col_list} FROM "{old_name}";'
		)
		
		conn.execute( f'DROP TABLE "{old_name}";' )
		conn.execute( f'ALTER TABLE "{temp_name}" RENAME TO "{new_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'ON "{old_name}"', f'ON "{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )

def rename_column( table_name: str, old_name: str, new_name: str ) -> None:
	"""
		Purpose:
		--------
		Rename a column within an existing SQLite table. Attempts native ALTER TABLE rename
		first; if it fails, falls back to a schema-safe rebuild preserving column order, data,
		and indexes.

		Parameters:
		-----------
		table_name : str
			Table containing the column.

		old_name : str
			Existing column name.

		new_name : str
			New column name.

		Returns:
		--------
		None
	"""
	if not table_name or not old_name or not new_name:
		return
	
	with create_connection( ) as conn:
		try:
			conn.execute(
				f'ALTER TABLE "{table_name}" RENAME COLUMN "{old_name}" TO "{new_name}";'
			)
			conn.commit( )
			return
		except Exception:
			pass
		
		row = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='table' AND name =?
			""",
			(table_name,)
		).fetchone( )
		
		if not row or not row[ 0 ]:
			raise ValueError( "Table definition not found." )
		
		create_sql = row[ 0 ]
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table_name,)
		).fetchall( )
		
		schema = conn.execute( f'PRAGMA table_info("{table_name}");' ).fetchall( )
		cols = [ r[ 1 ] for r in schema ]
		if old_name not in cols:
			raise ValueError( "Column not found." )
		
		mapped_cols = [ (new_name if c == old_name else c) for c in cols ]
		
		temp_table = f"{table_name}__rebuild_temp"
		
		col_defs: List[ str ] = [ ]
		pk_cols = [ r for r in schema if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		for row in schema:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			out_name = new_name if col_name == old_name else col_name
			col_def = f'"{out_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			col_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( col_defs )});'
		
		old_select = ", ".join( [ f'"{c}"' for c in cols ] )
		new_insert = ", ".join( [ f'"{c}"' for c in mapped_cols ] )
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		conn.execute(
			f'INSERT INTO "{temp_table}" ({new_insert}) SELECT {old_select} FROM "{table_name}";'
		)
		
		conn.execute( f'DROP TABLE "{table_name}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table_name}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql:
				idx_sql = idx_sql.replace( f'"{old_name}"', f'"{new_name}"' )
				conn.execute( idx_sql )
		
		conn.commit( )
		
def create_index( table: str, column: str ) -> None:
	"""
		Purpose:
		--------
		Create a safe SQLite index on a specified table column.
	
		Handles:
			- Spaces in column names
			- Special characters
			- Reserved words
			- Duplicate index names
			- Validation against actual table schema
	
		Parameters:
		-----------
		table : str
			Table name.
		column : str
			Column name to index.
	"""
	if not table or not column:
		return
	
	# ----------  Validate table exists
	tables = list_tables( )
	if table not in tables:
		raise ValueError( "Invalid table name." )
	
	# ----------  Validate column exists
	schema = create_schema( table )
	valid_columns = [ col[ 1 ] for col in schema ]
	
	if column not in valid_columns:
		raise ValueError( "Invalid column name." )
	
	# ----------  Sanitize index name (identifier only)
	safe_index_name = re.sub( r"[^0-9a-zA-Z_]+", "_", f"idx_{table}_{column}" )
	
	# ----------  Create index safely (quote identifiers)
	sql = f'CREATE INDEX IF NOT EXISTS "{safe_index_name}" ON "{table}"("{column}");'
	
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def apply_filters( df: pd.DataFrame ) -> pd.DataFrame:
	st.subheader( 'Advanced Filters' )
	col1, col2, col3 = st.columns( 3 )
	column = col1.selectbox( 'Column', df.columns )
	operator = col2.selectbox( 'Operator', [ '=', '!=', '>', '<', '>=', '<=', 'contains' ] )
	value = col3.text_input( 'Value' )
	if value:
		if operator == '=':
			df = df[ df[ column ] == value ]
		elif operator == '!=':
			df = df[ df[ column ] != value ]
		elif operator == '>':
			df = df[ df[ column ].astype( float ) > float( value ) ]
		elif operator == '<':
			df = df[ df[ column ].astype( float ) < float( value ) ]
		elif operator == '>=':
			df = df[ df[ column ].astype( float ) >= float( value ) ]
		elif operator == '<=':
			df = df[ df[ column ].astype( float ) <= float( value ) ]
		elif operator == 'contains':
			df = df[ df[ column ].astype( str ).str.contains( value ) ]
	
	return df

def create_aggregation( df: pd.DataFrame ):
	st.subheader( 'Aggregation Engine' )
	
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	
	if not numeric_cols:
		st.info( 'No numeric columns available.' )
		return
	
	col = st.selectbox( 'Column', numeric_cols )
	agg = st.selectbox( 'Aggregation', [ 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'MEDIAN' ] )
	
	if agg == 'COUNT':
		result = df[ col ].count( )
	elif agg == 'SUM':
		result = df[ col ].sum( )
	elif agg == 'AVG':
		result = df[ col ].mean( )
	elif agg == 'MIN':
		result = df[ col ].min( )
	elif agg == 'MAX':
		result = df[ col ].max( )
	elif agg == 'MEDIAN':
		result = df[ col ].median( )
	
	st.metric( 'Result', result )

def create_visualization( df: pd.DataFrame ):
	st.subheader( 'Visualization Engine' )
	numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
	categorical_cols = df.select_dtypes( include=[ 'object' ] ).columns.tolist( )
	chart = st.selectbox( 'Chart Type',
		[ 'Histogram', 'Bar', 'Line', 'Scatter', 'Box', 'Pie', 'Correlation' ] )
	
	if chart == 'Histogram' and numeric_cols:
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.histogram( df, x=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Bar':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.bar( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Line':
		x = st.selectbox( 'X', df.columns )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.line( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Scatter':
		x = st.selectbox( 'X', numeric_cols )
		y = st.selectbox( 'Y', numeric_cols )
		fig = px.scatter( df, x=x, y=y )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Box':
		col = st.selectbox( 'Column', numeric_cols )
		fig = px.box( df, y=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Pie':
		col = st.selectbox( 'Category Column', categorical_cols )
		fig = px.pie( df, names=col )
		st.plotly_chart( fig, use_container_width=True )
	
	elif chart == 'Correlation' and len( numeric_cols ) > 1:
		corr = df[ numeric_cols ].corr( )
		fig = px.imshow( corr, text_auto=True )
		st.plotly_chart( fig, use_container_width=True )

def dm_create_table_from_df( table_name: str, df: pd.DataFrame ):
	columns = [ ]
	for col in df.columns:
		sql_type = get_sqlite_type( df[ col ].dtype )
		safe_col = col.replace( ' ', '_' )
		columns.append( f'{safe_col} {sql_type}' )
	
	create_stmt = f'CREATE TABLE IF NOT EXISTS {table_name} ({", ".join( columns )});'
	
	with create_connection( ) as conn:
		conn.execute( create_stmt )
		conn.commit( )

def insert_data( table_name: str, df: pd.DataFrame ):
	df = df.copy( )
	df.columns = [ c.replace( ' ', '_' ) for c in df.columns ]
	
	placeholders = ', '.join( [ '?' ] * len( df.columns ) )
	stmt = f'INSERT INTO {table_name} VALUES ({placeholders});'
	
	with create_connection( ) as conn:
		conn.executemany( stmt, df.values.tolist( ) )
		conn.commit( )

def get_sqlite_type( dtype ) -> str:
	"""
		Purpose:
		--------
		Map a pandas dtype to an appropriate SQLite column type.
	
		Parameters:
		-----------
		dtype : pandas dtype
			The dtype of a pandas Series.
	
		Returns:
		--------
		str
			SQLite column type.
	"""
	dtype_str = str( dtype ).lower( )
	
	# ----------  Integer Types
	if "int" in dtype_str:
		return "INTEGER"
	
	# ----------  Float Types
	if "float" in dtype_str:
		return "REAL"
	
	# ----------  Boolean
	if "bool" in dtype_str:
		return "INTEGER"
	
	# ----------  Datetime
	if "datetime" in dtype_str:
		return "TEXT"
	
	# ----------  Categorical
	if "category" in dtype_str:
		return "TEXT"
	
	# ----------  Default fallback
	return "TEXT"

def create_custom_table( table_name: str, columns: list ) -> None:
	"""
		Purpose:
		--------
		Create a custom SQLite table from column definitions.
	
		Parameters:
		-----------
		table_name : str
			Name of table.
	
		columns : list of dict
			[
				{
					"name": str,
					"type": str,
					"not_null": bool,
					"primary_key": bool,
					"auto_increment": bool
				}
			]
	"""
	if not table_name:
		raise ValueError( "Table name required." )
	
	# ----------  Validate identifier
	if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", table_name ):
		raise ValueError( "Invalid table name." )
	
	col_defs = [ ]
	for col in columns:
		col_name = col[ "name" ]
		col_type = col[ "type" ].upper( )
		if not re.match( r"^[A-Za-z_][A-Za-z0-9_]*$", col_name ):
			raise ValueError( f"Invalid column name: {col_name}" )
		
		definition = f'"{col_name}" {col_type}'
		if col[ "primary_key" ]:
			definition += " PRIMARY KEY"
			if col[ "auto_increment" ] and col_type == "INTEGER":
				definition += " AUTOINCREMENT"
		
		if col[ "not_null" ]:
			definition += " NOT NULL"
	
		col_defs.append( definition )
	
	sql = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join( col_defs )});'
	with create_connection( ) as conn:
		conn.execute( sql )
		conn.commit( )

def is_safe_query( query: str ) -> bool:
	"""
	
		Purpose:
		--------
		Determine whether a SQL query is read-only and safe to execute.
	
		Allows:
			SELECT
			WITH (CTE returning SELECT)
			EXPLAIN SELECT
			PRAGMA (read-only)
	
		Blocks:
			INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, ATTACH,
			DETACH, VACUUM, REPLACE, TRIGGER, and multiple statements.
			
	"""
	if not query or not isinstance( query, str ):
		return False
	
	q = query.strip( ).lower( )
	
	# ----------  Block multiple statements
	if ';' in q[ :-1 ]:
		return False
	
	# ----------  Remove SQL comments
	q = re.sub( r"--.*?$", "", q, flags=re.MULTILINE )
	q = re.sub( r"/\*.*?\*/", "", q, flags=re.DOTALL )
	q = q.strip( )
	
	# ----------  Allowed starting keywords
	allowed_starts = ('select', 'with', 'explain', 'pragma')
	if not q.startswith( allowed_starts ):
		return False
	
	# ----------  Block dangerous keywords anywhere
	blocked_keywords = ('insert ', 'update ', 'delete ', 'drop ', 'alter ',
	                    'create ', 'attach ', 'detach ', 'vacuum ', 'replace ', 'trigger ')
	
	for keyword in blocked_keywords:
		if keyword in q:
			return False
	
	return True

def create_identifier( name: str ) -> str:
	"""
	
		Purpose:
		--------
		Sanitize a string into a safe SQLite identifier.
	
		- Replaces invalid characters with underscores
		- Ensures it starts with a letter or underscore
		- Prevents empty names
		
	"""
	if not name or not isinstance( name, str ):
		raise ValueError( 'Invalid Identifier.' )
	
	safe = re.sub( r'[^0-9a-zA-Z_]', '_', name.strip( ) )
	if not re.match( r'^[A-Za-z_]', safe ):
		safe = f'_{safe}'
	
	if not safe:
		raise ValueError( 'Invalid identifier after sanitization.' )
	
	return safe

def get_indexes( table: str ):
	with create_connection( ) as conn:
		rows = conn.execute( f'PRAGMA index_list("{table}");' ).fetchall( )
		return rows

def add_column( table: str, column: str, col_type: str ):
	column = create_identifier( column )
	col_type = col_type.upper( )
	
	with create_connection( ) as conn:
		conn.execute(
			f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type};' )
		conn.commit( )

def create_profile_table( table: str ):
	df = read_table( table )
	profile_rows = [ ]
	total_rows = len( df )
	for col in df.columns:
		series = df[ col ]
		null_count = series.isna( ).sum( )
		distinct_count = series.nunique( dropna=True )
		row = \
			{
					'column': col, 'dtype': str( series.dtype ),
					'null_%': round( (null_count / total_rows) * 100, 2 ) if total_rows else 0,
					'distinct_%': round( (
								                     distinct_count / total_rows) * 100, 2 ) if total_rows else 0,
			}
		
		if pd.api.types.is_numeric_dtype( series ):
			row[ "min" ] = series.min( )
			row[ "max" ] = series.max( )
			row[ "mean" ] = series.mean( )
		else:
			row[ "min" ] = None
			row[ "max" ] = None
			row[ "mean" ] = None
		
		profile_rows.append( row )
	
	return pd.DataFrame( profile_rows )

def drop_column( table: str, column: str ):
	if not table or not column:
		raise ValueError( "Table and column required." )
	
	with create_connection( ) as conn:
		schema = conn.execute( f'PRAGMA table_info("{table}");' ).fetchall( )
		if not schema:
			raise ValueError( "Table definition not found." )
		
		col_names = [ r[ 1 ] for r in schema ]
		if column not in col_names:
			raise ValueError( "Column not found." )
		
		remaining = [ r for r in schema if r[ 1 ] != column ]
		if not remaining:
			raise ValueError( "Cannot drop the only remaining column." )
		
		temp_table = f"{table}_rebuild_temp"
		
		pk_cols = [ r for r in remaining if int( r[ 5 ] or 0 ) > 0 ]
		single_pk = len( pk_cols ) == 1
		
		new_defs: List[ str ] = [ ]
		for row in remaining:
			col_name = row[ 1 ]
			col_type = row[ 2 ] or ''
			not_null = int( row[ 3 ] or 0 )
			default_value = row[ 4 ]
			pk = int( row[ 5 ] or 0 )
			
			col_def = f'"{col_name}" {col_type}'.strip( )
			
			if not_null:
				col_def += ' NOT NULL'
			
			if default_value is not None:
				col_def += f' DEFAULT {default_value}'
			
			if single_pk and pk == 1:
				col_def += ' PRIMARY KEY'
			
			new_defs.append( col_def )
		
		new_create_sql = f'CREATE TABLE "{temp_table}" ({", ".join( new_defs )});'
		
		conn.execute( "BEGIN" )
		conn.execute( new_create_sql )
		
		remaining_cols = [ r[ 1 ] for r in remaining ]
		col_list = ", ".join( [ f'"{c}"' for c in remaining_cols ] )
		
		conn.execute(
			f'INSERT INTO "{temp_table}" ({col_list}) '
			f'SELECT {col_list} FROM "{table}";'
		)
		
		indexes = conn.execute(
			"""
            SELECT sql
            FROM sqlite_master
            WHERE type ='index' AND tbl_name=? AND sql IS NOT NULL
			""",
			(table,)
		).fetchall( )
		
		conn.execute( f'DROP TABLE "{table}";' )
		conn.execute( f'ALTER TABLE "{temp_table}" RENAME TO "{table}";' )
		
		for idx in indexes:
			idx_sql = idx[ 0 ]
			if idx_sql and column not in idx_sql:
				conn.execute( idx_sql )
		
		conn.commit( )

# ------------- DOCQNA UTILITIES ----------------------

def extract_text_from_bytes( file_bytes: bytes ) -> str:
	"""
		Extracts text from PDF or text-based documents.
	"""
	try:
		import fitz  # PyMuPDF
		
		doc = fitz.open( stream=file_bytes, filetype="pdf" )
		text = ""
		for page in doc:
			text += page.get_text( )
		return text.strip( )
	
	except Exception:
		try:
			return file_bytes.decode( errors="ignore" )
		except Exception:
			return ""

def route_document_query( prompt: str ) -> str:
	"""
		Purpose:
		--------
		Route a document question through the unified chat pipeline and return a model-generated answer.

		Parameters:
		-----------
		prompt : str
			The user question to answer about active documents.

		Returns:
		--------
		str
			The assistant answer text.
	"""
	user_input = build_document_user_input( prompt )
	if not user_input:
		user_input = (prompt or '').strip( )
	
	return run_llm_turn(
		user_input=user_input,
		temperature=float( st.session_state.get( 'temperature', 0.0 ) ),
		top_p=float( st.session_state.get( 'top_percent', 0.95 ) ),
		repeat_penalty=float( st.session_state.get( 'repeat_penalty', 1.1 ) ),
		max_tokens=int( st.session_state.get( 'max_tokens', 1024 ) ) or 1024,
		stream=False,
		output=None
	)

def summarize_active_document( ) -> str:
	"""
		Uses the routing layer to summarize the currently active document.
	"""
	system_instructions = st.session_state.get( "system_instructions", "" )
	summary_prompt = """
		Provide a clear, structured summary of this document.
		Include:
		- Purpose
		- Key themes
		- Major conclusions
		- Important data points (if any)
		- Policy implications (if applicable)
		
		Be precise and concise.
		"""
	if system_instructions:
		summary_prompt = f"{system_instructions}\n\n{summary_prompt}"
	
	return route_document_query( summary_prompt.strip( ) )

def _docqna_compute_fingerprint( active_docs: List[ str ], doc_bytes: Dict[ str, bytes ] ) -> str:
	'''
		
		Purpose:
		--------
		Computes a stable fingerprint for the currently selected active documents and their byte contents.
	
		Parameters:
		-----------
		active_docs:
			A List[ str ] of active document names.
		doc_bytes:
			A Dict[ str, bytes ] mapping document name to file bytes.
	
		Returns:
		--------
		A str fingerprint suitable for cache invalidation.
	
	'''
	h = hashlib.sha256( )
	for name in sorted( active_docs ):
		b = doc_bytes.get( name, b'' )
		h.update( name.encode( 'utf-8', errors='ignore' ) )
		h.update( len( b ).to_bytes( 8, 'little', signed=False ) )
		h.update( hashlib.sha256( b ).digest( ) )
	return h.hexdigest( )

def _docqna_extract_text_from_pdf_bytes( file_bytes: bytes ) -> str:
	'''
	
		Purpose:
		--------
		Extracts text from a PDF byte stream using PyMuPDF.
	
		Parameters:
		-----------
		file_bytes:
			The PDF bytes.
	
		Returns:
		--------
		A str containing extracted text.
	
	'''
	if not file_bytes:
		return ''
	
	try:
		doc = fitz.open( stream=file_bytes, filetype='pdf' )
		parts: List[ str ] = [ ]
		for page in doc:
			parts.append( page.get_text( 'text' ) or '' )
		return '\n'.join( parts ).strip( )
	except Exception:
		return ''

def _docqna_safe_load_sqlite_vec( conn: sqlite3.Connection ) -> bool:
	'''
		
		Purpose:
		--------
		Attempts to load sqlite-vec into the provided SQLite connection.
	
		Parameters:
		-----------
		conn:
			The sqlite3.Connection.
	
		Returns:
		--------
		True if sqlite-vec loaded successfully; otherwise False.
		
	'''
	try:
		import sqlite_vec
		
		sqlite_vec.load( conn )
		return True
	except Exception:
		return False

def _docqna_ensure_vec_schema( dim: int ) -> bool:
	'''
	
		Purpose:
		--------
		Creates the sqlite-vec virtual table used for Document Q&A embeddings if possible.
	
		Parameters:
		-----------
		dim:
			The embedding dimension (e.g., 384 for all-MiniLM-L6-v2).
	
		Returns:
		--------
		True if the schema exists and is usable; otherwise False.
	
	'''
	conn = create_connection( )
	try:
		ok = _docqna_safe_load_sqlite_vec( conn )
		if not ok:
			return False
		
		cur = conn.cursor( )
		cur.execute(
			f'''
			CREATE VIRTUAL TABLE IF NOT EXISTS docqna_vec
			USING vec0(
				embedding float[{int( dim )}],
				doc_name TEXT,
				chunk TEXT
			);
			'''
		)
		conn.commit( )
		return True
	except Exception:
		return False
	finally:
		conn.close( )

def _docqna_rebuild_index_if_needed( embedder: SentenceTransformer ) -> None:
	'''
		
		Purpose:
		--------
		Builds or refreshes the Document Q&A vector index when active documents change.
	
		Parameters:
		-----------
		embedder:
			The SentenceTransformer used to generate embeddings.
	
		Returns:
		--------
		None
		
	'''
	active_docs: List[ str ] = st.session_state.get( 'active_docs', [ ] )
	doc_bytes: Dict[ str, bytes ] = st.session_state.get( 'doc_bytes', { } )
	
	fp = _docqna_compute_fingerprint( active_docs, doc_bytes )
	if fp and fp == st.session_state.get( 'docqna_fingerprint', '' ):
		return
	
	st.session_state[ 'docqna_fingerprint' ] = fp
	st.session_state[ 'docqna_chunk_count' ] = 0
	st.session_state[ 'docqna_fallback_rows' ] = [ ]
	
	dim_value = getattr( embedder, 'get_sentence_embedding_dimension', lambda: 384 )( )
	dim = int( dim_value ) if dim_value else 384
	
	vec_ready = _docqna_ensure_vec_schema( dim )
	st.session_state[ 'docqna_vec_ready' ] = bool( vec_ready )
	
	conn = create_connection( )
	try:
		cur = conn.cursor( )
		
		if vec_ready:
			try:
				cur.execute( 'DELETE FROM docqna_vec;' )
				conn.commit( )
			except Exception:
				st.session_state[ 'docqna_vec_ready' ] = False
				vec_ready = False
		
		total_chunks = 0
		fallback_rows: List[ Tuple[ str, str, bytes ] ] = [ ]
		
		for name in active_docs:
			b = doc_bytes.get( name )
			if not b:
				continue
			
			text = _docqna_extract_text_from_pdf_bytes( b )
			if not text:
				continue
			
			chunks = chunk_text( text )
			if not chunks:
				continue
			
			vecs = embedder.encode( chunks, show_progress_bar=False )
			vecs = np.asarray( vecs, dtype=np.float32 )
			
			if vec_ready:
				for chunk_text_value, v in zip( chunks, vecs ):
					cur.execute(
						'INSERT INTO docqna_vec ( embedding, doc_name, chunk ) VALUES ( ?, ?, ? );',
						(v.tobytes( ), name, chunk_text_value)
					)
			else:
				for chunk_text_value, v in zip( chunks, vecs ):
					fallback_rows.append( (name, chunk_text_value, v.tobytes( )) )
			
			total_chunks += int( len( chunks ) )
		
		conn.commit( )
		st.session_state[ 'docqna_chunk_count' ] = total_chunks
		
		if not vec_ready:
			st.session_state[ 'docqna_fallback_rows' ] = fallback_rows
	
	except Exception:
		st.session_state[ 'docqna_vec_ready' ] = False
		st.session_state[ 'docqna_fallback_rows' ] = [ ]
		st.session_state[ 'docqna_chunk_count' ] = 0
	finally:
		conn.close( )

def retrieve_top_doc_chunks( query: str, k: int = 6 ) -> List[ Tuple[ str, str, float ] ]:
	'''
	
		Purpose:
		--------
		Retrieves top-k document chunks relevant to the query, using sqlite-vec when available, and falling
		back to in-memory cosine similarity when not.
	
		Parameters:
		-----------
		query:
			The user query string.
		k:
			The number of chunks to return.
	
		Returns:
		--------
		A List[ Tuple[ str, str, float ] ] of (doc_name, chunk, score_or_distance).
	
	'''
	if not query or not query.strip( ):
		return [ ]
	
	embedder: SentenceTransformer = load_embedder( )
	_docqna_rebuild_index_if_needed( embedder )
	
	qv = embedder.encode( [ query ], show_progress_bar=False )
	qv = np.asarray( qv, dtype=np.float32 )[ 0 ]
	
	if st.session_state.get( 'docqna_vec_ready', False ):
		conn = create_connection( )
		try:
			_docqna_safe_load_sqlite_vec( conn )
			cur = conn.cursor( )
			cur.execute(
				'''
                SELECT doc_name, chunk, distance
                FROM docqna_vec
                WHERE embedding MATCH ?
                ORDER BY distance ASC LIMIT ?;
				''',
				(qv.tobytes( ), int( k ))
			)
			rows = cur.fetchall( )
			return [ (r[ 0 ], r[ 1 ], float( r[ 2 ] )) for r in rows ]
		except Exception:
			st.session_state[ 'docqna_vec_ready' ] = False
		finally:
			conn.close( )
	
	fallback_rows: List[
		Tuple[ str, str, bytes ] ] = st.session_state.get( 'docqna_fallback_rows', [ ] )
	results: List[ Tuple[ str, str, float ] ] = [ ]
	
	for doc_name, chunk_text_value, vec_blob in fallback_rows:
		if not vec_blob:
			continue
		
		v = np.frombuffer( vec_blob, dtype=np.float32 )
		if v.size == 0:
			continue
		
		score = cosine_similarity( qv, v )
		results.append( (doc_name, chunk_text_value, float( score )) )
	
	results.sort( key=lambda r: r[ 2 ], reverse=True )
	return results[ : int( k ) ]

def build_document_user_input( user_query: str, k: int = 6 ) -> str:
	'''
	
		Purpose:
		--------
		Builds a Document Q&A prompt that injects retrieved chunks (RAG) instead of stuffing full documents.
	
		Parameters:
		-----------
		user_query:
			The user question.
		k:
			The number of retrieved chunks to include.
	
		Returns:
		--------
		A str prompt suitable for llama.cpp completion.
	
	'''
	system = str( st.session_state.get( 'system_instructions', '' ) or '' ).strip( )
	hits = retrieve_top_doc_chunks( user_query, k=int( k ) )
	
	context_blocks: List[ str ] = [ ]
	for doc_name, chunk, score in hits:
		context_blocks.append( f'[Document: {doc_name}]\n{chunk}'.strip( ) )
	
	context = '\n\n'.join( context_blocks ).strip( )
	
	prompt_parts: List[ str ] = [ ]
	
	if system:
		prompt_parts.append( system )
	
	if context:
		prompt_parts.append(
			'Use the following document excerpts to answer the question. If the excerpts do not contain '
			'the answer, say you do not have enough information.\n\n'
			f'{context}'
		)
	
	prompt_parts.append( f'Question:\n{user_query}\n\nAnswer:' )
	
	return '\n\n'.join( prompt_parts ).strip( )

# -------------- LLM  UTILITIES -------------------

@st.cache_resource
def load_llm( ctx: int, threads: int ) -> Llama:
	return Llama( model_path=str( MODEL_PATH_OBJ ), n_ctx=ctx, n_threads=threads, n_batch=512,
		verbose=False )

@st.cache_resource
def load_embedder( ) -> SentenceTransformer:
	return SentenceTransformer( 'all-MiniLM-L6-v2' )

# ==============================================================================
# Init
# ==============================================================================
initialize_database( )
llm = load_llm( cfg.DEFAULT_CTX, cfg.CORES )
embedder = load_embedder( )

if not isinstance( st.session_state.get( 'messages' ), list ):
	st.session_state[ 'messages' ] = [ ]

if len( st.session_state[ 'messages' ] ) == 0:
	st.session_state[ 'messages' ] = load_history( )

if 'system_instructions' not in st.session_state:
	st.session_state[ 'system_instructions' ] = ''

st.set_page_config( page_title='Bro', layout='wide', page_icon=cfg.FAVICON )

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
	style_subheaders( )
	st.logo( cfg.LOGO, size='large' )
	
	c1, c2 = st.columns( [ 0.05, 0.95] )
	with c2:
		st.subheader( '⚙️ Application Mode' )
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		mode = st.radio( label='', options=cfg.MODES, index=0 )
	
	st.divider( )
	
# ==============================================================================
# TEXT GENERATION MODE
# ==============================================================================
if mode == 'Text Generation':
	st.subheader( "💬 Text Generation", help=cfg.TEXT_GENERATION )
	st.divider( )
	messages = st.session_state.get( 'messages', [ ] )
	max_tokens = st.session_state.get( 'max_tokens', 0 )
	top_percent = st.session_state.get( 'top_percent', 0.0 )
	top_k = st.session_state.get( 'top_k', 0 )
	temperature = st.session_state.get( 'temperature', 0.0 )
	is_grounded = st.session_state.get( 'is_grounded', False )
	frequency_penalty = st.session_state.get( 'frequency_penalty', 0.0 )
	presense_penalty = st.session_state.get( 'presense_penalty', 0.0 )
	repeat_penalty = st.session_state.get( 'repeat_penalty', 0.0 )
	repeat_window = st.session_state.get( 'repeat_window', 0.0 )
	cpu_threads = st.session_state.get( 'cpu_threads', cfg.CORES )
	context_window = st.session_state.get( 'context_window', cfg.DEFAULT_CTX )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander — Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False ):
			with st.expander( label='⚙️ Response Controls', expanded=False ):
				mind_c1, mind_c2, mind_c3, mind_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium' )
				
				# ------------- Temperature ----------
				with mind_c1:
					set_temperature = st.slider( label='Temperature', min_value=0.0, max_value=1.0,
						help=cfg.TEMPERATURE, key='temperature' )
				
					temperature = st.session_state[ 'temperature' ]
					
				# ------------- Top-P ----------
				with mind_c2:
					set_top_p = st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						step=0.01, key='top_percent', help=cfg.TOP_P )
					
					top_percent = st.session_state[ 'top_percent' ]
				
				# ------------- Top-K ----------
				with mind_c3:
					set_top_k = st.slider( label='Top-K', min_value=0, max_value=50, step=1,
						key='top_k', help=cfg.TOP_K )
					
					top_k = st.session_state[ 'top_k' ]
				
				# ------------ Grounding --------
				with mind_c4:
					set_grounding = st.toggle( label='Use Grounding', value=False,
						key='is_grounded' )
					
					is_grounded = st.session_state[ 'is_grounded' ]
					
				# ------------- Reset Settings ----------
				if st.button( label='Reset', key='response_controls_reset', width='stretch' ):
					for key in [ 'top_k', 'top_percent', 'temperature', 'is_grounded' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
			
			# ------------------------------------------------------------------
			# Expander — Probability Controls
			# ------------------------------------------------------------------
			with st.expander( label='🎚️ Probability Controls', expanded=False ):
				prob_c1, prob_c2, prob_c3, prob_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium' )
				
				# ------------- Repeat Window ----------
				with prob_c1:
					set_repeat_last_n = st.slider( label='Repeat Window', min_value=0, max_value=1024,
						step=16, key='repeat_window', help=cfg.REPEAT_WINDOW )
					
					repeat_window = st.session_state[ 'repeat_window' ]
				
				# ------------- Repeat Penalty ----------
				with prob_c2:
					set_repeat_penalty = st.slider( label='Repeat Penalty', min_value=0.0, max_value=2.0,
						key='repeat_penalty', step=0.05, help=cfg.REPEAT_PENALTY )
					
					repeat_penalty = st.session_state[ 'repeat_penalty' ]
				
				# ------------- Presense Penalty ----------
				with prob_c3:
					set_presence_penalty = st.slider( label='Presence Penalty', min_value=0.0, max_value=2.0,
						key='presense_penalty', step=0.05, help=cfg.PRESENCE_PENALTY )
					
					presense_penalty = st.session_state[ 'presense_penalty' ]
				
				# ------------- Frequency Penalty ----------
				with prob_c4:
					set_frequency_penalty = st.slider( label='Frequency Penalty', min_value=0.0, max_value=2.0,
						key='frequency_penalty', step=0.05, help=cfg.FREQUENCY_PENALTY )
					
					frequency_penalty = st.session_state[ 'frequency_penalty' ]
				
				# ------------- Reset Settings ----------
				if st.button( label='Reset', key='probability_controls_reset', width='stretch' ):
					for key in [ 'frequency_penalty', 'presense_penalty',
					             'temperature', 'repeat_penalty', 'repeat_window' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
			
			# ------------------------------------------------------------------
			# Expander — Context Controls
			# ------------------------------------------------------------------
			with st.expander( label='🎛️ Context Controls', expanded=False ):
				ctx_c1, ctx_c2, ctx_c3, ctx_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium' )
				
				# ------------- Context Window ----------
				with ctx_c1:
					set_ctx = st.slider( label='Context Window', min_value=0, max_value=8192,
						key='context_window', step=512, help=cfg.CONTEXT_WINDOW )
					
					context_window = st.session_state[ 'context_window' ]
				
				# ------------- CPU Threads ----------
				with ctx_c2:
					set_threads = st.slider( label='CPU Threads', min_value=0, max_value=cfg.CORES,
						key='cpu_threads', step=1, help=cfg.CPU_CORES, )
					
					threads = st.session_state[ 'cpu_threads' ]
				
				# ------------- Max Tokens ----------
				with ctx_c3:
					set_max_tokens = st.slider( label='Max Tokens', min_value=0, max_value=4096, step=128,
						key='max_tokens', help=cfg.MAX_TOKENS, )
				
				# ------------- Random Seed ----------
				with ctx_c4:
					set_seed = st.slider( label="Random Seed", min_value=0, max_value=4096, step=1,
						key='random_seed', help=cfg.SEED )
				
				# ------------- Reset Settings ----------
				if st.button( label='Reset', key='context_controls_reset', width='stretch' ):
					for key in [ 'random_seed', 'max_tokens', 'cpu_threads', 'context_window' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			ins_left, ins_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			with ins_left:
				st.text_area( label='Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'system_instructions' ] = text
			
			with ins_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=_on_template_change )
			
			def _on_clear( ) -> None:
				st.session_state[ 'system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( label='Clear Instructions', width='stretch', on_click=_on_clear )
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		for r, c in st.session_state.messages:
			with st.chat_message( r ):
				st.markdown( c )
			
		user_input = st.chat_input( 'Ask Bro…' )
		if user_input:
			save_message( 'user', user_input )
			st.session_state.messages.append( ('user', user_input) )
			
			with st.chat_message( 'user' ):
				st.markdown( user_input )
			
			with st.chat_message( 'assistant' ):
				out = st.empty( )
				buf = run_llm_turn(
					user_input=user_input,
					temperature=float( temperature ),
					top_p=float( top_percent ),
					repeat_penalty=float( repeat_penalty ),
					max_tokens=1024,
					stream=True,
					output=out
				)
			
			save_message( 'assistant', buf )
			st.session_state.messages.append( ('assistant', buf) )
		
		if st.button( '🧹 Clear Chat' ):
			clear_history( )
			st.session_state.messages = [ ]
			st.rerun( )
		
# ==============================================================================
# RETRIEVAL AUGMENTATION
# ==============================================================================
elif mode == 'Document Q&A':
	st.subheader( "📚 Retrieval Augementation", help=cfg.RETRIEVAL_AUGMENTATION )
	st.divider( )
	messages = st.session_state.get( 'messages', [ ] )
	uploaded = st.session_state.get( 'uploaded', [ ] )
	active_docs = st.session_state.get( 'active_docs', [ ] )
	doc_bytes = st.session_state.get( 'doc_bytes', { } )
	max_tokens = st.session_state.get( 'max_tokens', 0 )
	top_percent = st.session_state.get( 'top_percent', 0.0 )
	top_k = st.session_state.get( 'top_k', 0 )
	temperature = st.session_state.get( 'temperature', 0.0 )
	frequency_penalty = st.session_state.get( 'frequency_penalty', 0.0 )
	presense_penalty = st.session_state.get( 'presense_penalty', 0.0 )
	repeat_penalty = st.session_state.get( 'repeat_penalty', 0.0 )
	repeat_window = st.session_state.get( 'repeat_window', 0.0 )
	cpu_threads = st.session_state.get( 'cpu_threads', cfg.CORES )
	context_window = st.session_state.get( 'context_window', cfg.DEFAULT_CTX )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		# ------------------------------------------------------------------
		# Expander — Mind Controls
		# ------------------------------------------------------------------
		with st.expander( label='Mind Controls', icon='🧠', expanded=False ):
			with st.expander( label='⚙️ Response Controls', expanded=False ):
				mind_c1, mind_c2, mind_c3 = st.columns( [ .33, .33, .33 ], border=True, gap='medium' )
				
				# ------------- Temperature ----------
				with mind_c1:
					set_temperature = st.slider( label='Temperature', min_value=0.0, max_value=1.0,
						value=float( st.session_state.get( 'temperature' ) ),
						help=cfg.TEMPERATURE, key='temperature' )
					
					temperature = st.session_state[ 'temperature' ]
				
				# ------------- Top-P ----------
				with mind_c2:
					set_top_p = st.slider( label='Top-P', min_value=0.0, max_value=1.0,
						step=0.01, key='top_percent', help=cfg.TOP_P )
					
					top_percent = st.session_state[ 'top_percent' ]
				
				# ------------- Top-K ----------
				with mind_c3:
					set_top_k = st.slider( label='Top-K', min_value=0, max_value=50, step=1,
						key='top_k', help=cfg.TOP_K )
					
					top_k = st.session_state[ 'top_k' ]
				
				# ------------- Reset Settings ----------
				if st.button( label='Reset', key='response_controls_reset', width='stretch' ):
					for key in [ 'top_k', 'top_percent', 'temperature' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
			
			# ------------------------------------------------------------------
			# Expander — Probability Controls
			# ------------------------------------------------------------------
			with st.expander( label='🎚️ Probability Controls', expanded=False ):
				prob_c1, prob_c2, prob_c3, prob_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium' )
				
				# ------------- Repeat Window ----------
				with prob_c1:
					set_repeat_last_n = st.slider( label='Repeat Window', min_value=0, max_value=1024,
						step=16, key='repeat_window', help=cfg.REPEAT_WINDOW )
					
					repeat_window = st.session_state[ 'repeat_window' ]
				
				# ------------- Repeat Penalty ----------
				with prob_c2:
					set_repeat_penalty = st.slider( label='Repeat Penalty', min_value=0.0, max_value=2.0,
						key='repeat_penalty', step=0.05, help=cfg.REPEAT_PENALTY )
					
					repeat_penalty = st.session_state[ 'repeat_penalty' ]
				
				# ------------- Presense Penalty ----------
				with prob_c3:
					set_presence_penalty = st.slider( label='Presence Penalty', min_value=0.0, max_value=2.0,
						key='presense_penalty', step=0.05, help=cfg.PRESENCE_PENALTY )
					
					presense_penalty = st.session_state[ 'presense_penalty' ]
				
				# ------------- Frequency Penalty ----------
				with prob_c4:
					set_frequency_penalty = st.slider( label='Frequency Penalty', min_value=0.0, max_value=2.0,
						key='frequency_penalty', step=0.05, help=cfg.FREQUENCY_PENALTY )
					
					frequency_penalty = st.session_state[ 'frequency_penalty' ]
				
				# ------------- Reset Settings ----------
				if st.button( label='Reset', key='probability_controls_reset', width='stretch' ):
					for key in [ 'frequency_penalty', 'presense_penalty',
					             'temperature', 'repeat_penalty', 'repeat_window' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
			
			# ------------------------------------------------------------------
			# Expander — Context Controls
			# ------------------------------------------------------------------
			with st.expander( label='🎛️ Context Controls', expanded=False ):
				ctx_c1, ctx_c2, ctx_c3, ctx_c4 = st.columns( [ 0.25, 0.25, 0.25, 0.25 ],
					border=True, gap='medium' )
				
				# ------------- Context Window ----------
				with ctx_c1:
					set_ctx = st.slider( label='Context Window', min_value=0, max_value=8192,
						key='context_window', step=512, help=cfg.CONTEXT_WINDOW )
					
					context_window = st.session_state[ 'context_window' ]
				
				# ------------- CPU Threads ----------
				with ctx_c2:
					set_threads = st.slider( label='CPU Threads', min_value=0, max_value=cfg.CORES,
						key='cpu_threads', step=1, help=cfg.CPU_CORES, )
					
					threads = st.session_state[ 'cpu_threads' ]
				
				# ------------- Max Tokens ----------
				with ctx_c3:
					set_max_tokens = st.slider( label='Max Tokens', min_value=0, max_value=4096, step=128,
						key='max_tokens', help=cfg.MAX_TOKENS, )
				
				# ------------- Random Seed ----------
				with ctx_c4:
					set_seed = st.slider( label="Random Seed", min_value=0, max_value=4096, step=1,
						key='random_seed', help=cfg.SEED )
				
				# ------------- Reset Settings ----------
				if st.button( label='Reset', key='context_controls_reset', width='stretch' ):
					for key in [ 'random_seed', 'max_tokens', 'cpu_threads', 'context_window' ]:
						if key in st.session_state:
							del st.session_state[ key ]
					
					st.rerun( )
		
		# ------------------------------------------------------------------
		# Expander — System Instructions
		# ------------------------------------------------------------------
		with st.expander( label='System Instructions', icon='🖥️', expanded=False, width='stretch' ):
			ins_left, ins_right = st.columns( [ 0.8, 0.2 ] )
			prompt_names = fetch_prompt_names( cfg.DB_PATH )
			if not prompt_names:
				prompt_names = [ '' ]
			
			with ins_left:
				st.text_area( label='Enter Text', height=50, width='stretch',
					help=cfg.SYSTEM_INSTRUCTIONS, key='system_instructions' )
			
			def _on_template_change( ) -> None:
				name = st.session_state.get( 'instructions' )
				if name and name != 'No Templates Found':
					text = fetch_prompt_text( cfg.DB_PATH, name )
					if text is not None:
						st.session_state[ 'system_instructions' ] = text
			
			with ins_right:
				st.selectbox( label='Use Template', options=prompt_names, index=None,
					key='instructions', on_change=_on_template_change )
			
			def _on_clear( ) -> None:
				st.session_state[ 'system_instructions' ] = ''
				st.session_state[ 'instructions' ] = ''
			
			st.button( label='Clear Instructions', width='stretch', on_click=_on_clear )
		
		# ------------------------------------------------------------------
		# Document Selection UI
		# ------------------------------------------------------------------
		with st.expander( label='Document Loader', icon='📥', expanded=False, width='stretch' ):
			doc_left, doc_right = st.columns( [ 0.5, 0.5 ], gap='medium', border=True )
			with doc_left:
				doc_source = st.radio(
					label='Document Source',
					options=[ 'uploadlocal' ],
					index=0,
					horizontal=True,
					key='doc_source'
				)
				
				uploaded = st.file_uploader(
					label='Upload a document (PDF, TXT, DOCX)',
					type=[ 'pdf', 'txt', 'docx' ],
					accept_multiple_files=True,
					label_visibility='visible'
				)
				
				if uploaded is not None and type( uploaded ) == list and len( uploaded ) > 0:
					st.session_state.uploaded = uploaded
					
					names: List[ str ] = [ f.name for f in uploaded if getattr( f, 'name', None ) ]
					st.session_state.active_docs = names
					
					if 'doc_bytes' not in st.session_state or not isinstance( st.session_state.doc_bytes, dict ):
						st.session_state.doc_bytes = { }
					
					for f in uploaded:
						try:
							if getattr( f, 'name', None ):
								st.session_state.doc_bytes[ f.name ] = f.getvalue( )
						except Exception:
							continue
				else:
					st.info( 'Load a document.' )
				
				unload = st.button( label='Unload Document', width='stretch' )
				if unload:
					st.session_state.uploaded = [ ]
					st.session_state.active_docs = [ ]
					st.session_state.doc_bytes = { }
			
			with doc_right:
				if st.session_state.get( 'active_docs' ):
					name = st.session_state.active_docs[ 0 ]
					file_bytes = st.session_state.doc_bytes.get( name )
					
					if file_bytes:
						st.pdf( file_bytes, height=420 )
					else:
						st.info( "Document loaded but preview unavailable." )
				else:
					st.info( "No document loaded." )
			
		# ------------------------------------------------------------------
		# Chat History Render
		# ------------------------------------------------------------------
		if 'messages' not in st.session_state or not isinstance( st.session_state.messages, list ):
			st.session_state.messages = [ ]
		
		for msg in st.session_state.messages:
			role = ''
			content = ''
			
			if isinstance( msg, dict ):
				role = str( msg.get( 'role', '' ) or '' ).strip( )
				content = msg.get( 'content', '' )
			else:
				if isinstance( msg, tuple ) or isinstance( msg, list ):
					if len( msg ) == 2:
						role = str( msg[ 0 ] or '' ).strip( )
						content = msg[ 1 ]
					else:
						role = ''
						content = ''
				else:
					role = ''
					content = ''
					
			if role not in ('user', 'assistant', 'system'):
				continue
			
			if content is None:
				content = ''
			elif not isinstance( content, str ):
				content = str( content )
			
			with st.chat_message( role ):
				st.markdown( content )
				
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Chat Input
		# ------------------------------------------------------------------
		user_input = st.chat_input( 'Ask a question about the document' )
		if user_input and isinstance( user_input, str ) and user_input.strip( ):
			user_input = user_input.strip( )
			
			if 'messages' not in st.session_state or not isinstance( st.session_state.messages, list ):
				st.session_state.messages = [ ]
			
			save_message( 'user', user_input )
			st.session_state.messages.append( ('user', user_input) )
			
			with st.chat_message( 'user' ):
				st.markdown( user_input )
			
			doc_user_input = build_document_user_input( user_input )
			if not doc_user_input or not isinstance( doc_user_input, str ) or not doc_user_input.strip( ):
				doc_user_input = user_input
			
			with st.chat_message( 'assistant' ):
				out = st.empty( )
				response = run_llm_turn(
					user_input=doc_user_input,
					temperature=float( st.session_state.get( 'temperature', 0.0 ) ),
					top_p=float( st.session_state.get( 'top_percent', 0.95 ) ),
					repeat_penalty=float( st.session_state.get( 'repeat_penalty', 1.1 ) ),
					max_tokens=int( st.session_state.get( 'max_tokens', 1024 ) ) or 1024,
					stream=True,
					output=out )
			
			if response is None:
				response = ''
			elif not isinstance( response, str ):
				response = str( response )
			
			response = response.strip( )
			save_message( 'assistant', response )
			st.session_state.messages.append( ('assistant', response) )
			
# ==============================================================================
# SEMANTIC SEARCH
# ==============================================================================
elif mode == 'Semantic Search':
	st.subheader( "🔍 Semantic Search", help=cfg.SEMANTIC_SEARCH )
	st.divider( )
	
	# ------------------------------------------------------------------
	# Main Chat UI
	# ------------------------------------------------------------------
	left, center, right = st.columns( [ 0.05, 0.9, 0.05 ] )
	with center:
		st.session_state.use_semantic = st.checkbox( 'Use Semantic Context',
			st.session_state.use_semantic )
		files = st.file_uploader( 'Upload for embedding', accept_multiple_files=True )
		if files:
			chunks = [ ]
			for f in files:
				chunks.extend( chunk_text( f.read( ).decode( errors='ignore' ) ) )
			vecs = embedder.encode( chunks )
			with sqlite3.connect( cfg.DB_PATH ) as conn:
				conn.execute( 'DELETE FROM embeddings' )
				for c, v in zip( chunks, vecs ):
					conn.execute(
						'INSERT INTO embeddings (chunk, vector) VALUES (?, ?)',
						( c, v.tobytes( ) ) )
			st.success( 'Semantic index built' )

# ==============================================================================
# PROMPT ENGINEERING MODE
# ==============================================================================
elif mode == 'Prompt Engineering':
	st.subheader( '📝 Prompt Engineering', help=cfg.PROMPT_ENGINEERING )
	st.divider( )
	import sqlite3
	import math
	
	TABLE = 'Prompts'
	PAGE_SIZE = 10
	st.session_state.setdefault( 'pe_cascade_enabled', False )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		st.checkbox( 'Cascade selection into System Instructions', key='pe_cascade_enabled' )
		
		# ------------------------------------------------------------------
		# Session state
		# ------------------------------------------------------------------
		st.session_state.setdefault( 'pe_page', 1 )
		st.session_state.setdefault( 'pe_search', '' )
		st.session_state.setdefault( 'pe_sort_col', 'PromptsId' )
		st.session_state.setdefault( 'pe_sort_dir', 'ASC' )
		st.session_state.setdefault( 'pe_selected_id', None )
		st.session_state.setdefault( 'pe_caption', '' )
		st.session_state.setdefault( 'pe_name', '' )
		st.session_state.setdefault( 'pe_text', '' )
		st.session_state.setdefault( 'pe_version', '' )
		st.session_state.setdefault( 'pe_id', 0 )
		
		# ------------------------------------------------------------------
		# DB helpers
		# ------------------------------------------------------------------
		def get_conn( ):
			return sqlite3.connect( cfg.DB_PATH )
		
		def reset_selection( ):
			st.session_state.pe_selected_id = None
			st.session_state.pe_caption = ''
			st.session_state.pe_name = ''
			st.session_state.pe_text = ''
			st.session_state.pe_version = ''
			st.session_state.pe_id = 0
		
		def load_prompt( pid: int ) -> None:
			with get_conn( ) as conn:
				_select = f"SELECT Caption, Name, Text, Version, ID FROM {TABLE} WHERE PromptsId=?"
				cur = conn.execute( _select, (pid,), )
				row = cur.fetchone( )
				if not row:
					return
				st.session_state.pe_caption = row[ 0 ]
				st.session_state.pe_name = row[ 1 ]
				st.session_state.pe_text = row[ 2 ]
				st.session_state.pe_version = row[ 3 ]
				st.session_state.pe_id = row[ 4 ]
		
		# ------------------------------------------------------------------
		# Filters
		# ------------------------------------------------------------------
		c1, c2, c3, c4 = st.columns( [ 4, 2, 2, 3 ] )
		with c1:
			st.text_input( 'Search (Name/Text contains)', key='pe_search' )
		
		with c2:
			st.selectbox( 'Sort by', [ 'PromptsId', 'Caption', 'Name', 'Text', 'Version', 'ID' ],
				key='pe_sort_col', )
		
		with c3:
			st.selectbox( 'Direction', [ 'ASC', 'DESC' ], key='pe_sort_dir' )
		
		with c4:
			st.markdown(
				"<div style='font-size:0.95rem;font-weight:600;margin-bottom:0.25rem;'>Go to ID</div>",
				unsafe_allow_html=True, )
			
			a1, a2, a3 = st.columns( [ 2, 1, 1 ] )
			with a1:
				jump_id = st.number_input( 'Go to ID', min_value=1,
					step=1, label_visibility='collapsed', )
			
			with a2:
				if st.button( 'Go' ):
					st.session_state.pe_selected_id = int( jump_id )
					load_prompt( int( jump_id ) )
			
			with a3:
				st.button( 'Clear', on_click=reset_selection )
		
		# ------------------------------------------------------------------
		# Load prompt table
		# ------------------------------------------------------------------
		where = ""
		params = [ ]
		if st.session_state.pe_search:
			where = 'WHERE Name LIKE ? OR Text LIKE ?'
			s = f"%{st.session_state.pe_search}%"
			params.extend( [ s, s ] )
		
		offset = (st.session_state.pe_page - 1) * PAGE_SIZE
		query = f"""
	        SELECT PromptsId, Caption, Name, Text, Version, ID
	        FROM {TABLE}
	        {where}
	        ORDER BY {st.session_state.pe_sort_col} {st.session_state.pe_sort_dir}
	        LIMIT {PAGE_SIZE} OFFSET {offset}
	    """
		
		count_query = f"SELECT COUNT(*) FROM {TABLE} {where}"
		
		with get_conn( ) as conn:
			rows = conn.execute( query, params ).fetchall( )
			total_rows = conn.execute( count_query, params ).fetchone( )[ 0 ]
		
		total_pages = max( 1, math.ceil( total_rows / PAGE_SIZE ) )
		
		# ------------------------------------------------------------------
		# Prompt table
		# ------------------------------------------------------------------
		table_rows = [ ]
		for r in rows:
			table_rows.append(
				{
						'Selected': r[ 0 ] == st.session_state.pe_selected_id,
						'PromptsId': r[ 0 ],
						'Caption': r[ 1 ],
						'Name': r[ 2 ],
						'Text': r[ 3 ],
						'Version': r[ 4 ],
						'ID': r[ 5 ],
				} )
		
		edited = st.data_editor( table_rows, hide_index=True, use_container_width=True,
			key="prompt_table", )
		
		# ------------------------------------------------------------------
		# SELECTION PROCESSING (must run BEFORE widgets below)
		# ------------------------------------------------------------------
		selected = [ r for r in edited if isinstance( r, dict ) and r.get( 'Selected' ) ]
		if len( selected ) == 1:
			pid = int( selected[ 0 ][ 'PromptsId' ] )
			if pid != st.session_state.pe_selected_id:
				st.session_state.pe_selected_id = pid
				load_prompt( pid )
		
		elif len( selected ) == 0:
			reset_selection( )
		
		elif len( selected ) > 1:
			st.warning( 'Select exactly one prompt row.' )
		
		# ------------------------------------------------------------------
		# Paging
		# ------------------------------------------------------------------
		p1, p2, p3 = st.columns( [ 0.25, 3.5, 0.25 ] )
		with p1:
			if st.button( "◀ Prev" ) and st.session_state.pe_page > 1:
				st.session_state.pe_page -= 1
		
		with p2:
			st.markdown( f"Page **{st.session_state.pe_page}** of **{total_pages}**" )
		
		with p3:
			if st.button( "Next ▶" ) and st.session_state.pe_page < total_pages:
				st.session_state.pe_page += 1
		
		st.markdown( cfg.BLUE_DIVIDER, unsafe_allow_html=True )
		
		# ------------------------------------------------------------------
		# Edit Prompt
		# ------------------------------------------------------------------
		with st.expander( "🖊️ Edit Prompt", expanded=False ):
			st.text_input( "PromptsId", value=st.session_state.pe_selected_id or "",
				disabled=True )
			st.text_input( 'Caption', key='pe_caption' )
			st.text_input( 'Name', key='pe_name' )
			st.text_area( 'Text', key='pe_text', height=260 )
			st.text_input( 'Version', key='pe_version' )
			
			
			c1, c2, c3 = st.columns( 3 )
			with c1:
				save_label = '💾 Save Changes' if st.session_state.pe_selected_id else '➕ Create Prompt'
				if st.button( save_label ):
					with get_conn( ) as conn:
						if st.session_state.pe_selected_id:
							conn.execute(
								f"""
	                            UPDATE {TABLE}
	                            SET Caption=?, Name=?, Text=?, Version=?, ID=?
	                            WHERE PromptsId=?
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id,
										st.session_state.pe_selected_id
								), )
						else:
							conn.execute(
								f"""
	                            INSERT INTO {TABLE} (Caption, Name, Text, Version, ID)
	                            VALUES (?, ?, ?, ? , ?)
	                            """,
								(
										st.session_state.pe_caption,
										st.session_state.pe_name,
										st.session_state.pe_text,
										st.session_state.pe_version,
										st.session_state.pe_id
								),
							)
						conn.commit( )
					
					st.success( 'Saved.' )
					reset_selection( )
			
			with c2:
				if st.session_state.pe_selected_id and st.button( 'Delete' ):
					with get_conn( ) as conn:
						conn.execute(
							f'DELETE FROM {TABLE} WHERE PromptsId=?',
							(st.session_state.pe_selected_id,), )
						conn.commit( )
					reset_selection( )
					st.success( 'Deleted.' )
			
			with c3:
				st.button( '🧹 Clear Selection', on_click=reset_selection )

# ==============================================================================
# DATA MANAGEMENT MODE
# ==============================================================================
elif mode == 'Data Management':
	st.subheader( "🏛️ Data Management", help=cfg.DATA_MANAGEMENT )
	st.divider( )
	left, center, right = st.columns( [ 0.05, 0.90, 0.05 ] )
	with center:
		tabs = st.tabs( [ "📥 Import", "🗂 Browse", "💉 CRUD", "📊 Explore", "🔎 Filter",
		                  "🧮 Aggregate", "📈 Visualize", "⚙ Admin", "🧠 SQL" ] )
		
		tables = list_tables( )
		if not tables:
			st.info( "No tables available." )
		else:
			table = st.selectbox( "Table", tables )
			df_full = read_table( table )
		
		# ------------------------------------------------------------------------------
		# UPLOAD TAB
		# ------------------------------------------------------------------------------
		with tabs[ 0 ]:
			uploaded_file = st.file_uploader( 'Upload Excel File', type=[ 'xlsx' ] )
			overwrite = st.checkbox( 'Overwrite existing tables', value=True )
			if uploaded_file:
				try:
					sheets = pd.read_excel( uploaded_file, sheet_name=None )
					with create_connection( ) as conn:
						conn.execute( 'BEGIN' )
						for sheet_name, df in sheets.items( ):
							table_name = create_identifier( sheet_name )
							if overwrite:
								conn.execute( f'DROP TABLE IF EXISTS "{table_name}"' )
							
							# --- Create Table ---
							columns = [ ]
							df.columns = [ create_identifier( c ) for c in df.columns ]
							for col in df.columns:
								sql_type = get_sqlite_type( df[ col ].dtype )
								columns.append( f'"{col}" {sql_type}' )
							
							create_stmt = (
									f'CREATE TABLE "{table_name}" '
									f'({", ".join( columns )});'
							)
							
							conn.execute( create_stmt )
							
							# --- Insert Data ---
							placeholders = ", ".join( [ "?" ] * len( df.columns ) )
							insert_stmt = (
									f'INSERT INTO "{table_name}" '
									f'VALUES ({placeholders});'
							)
							
							conn.executemany(
								insert_stmt,
								df.where( pd.notnull( df ), None ).values.tolist( )
							)
						
						conn.commit( )
					
					st.success( 'Import completed successfully (transaction committed).' )
					st.rerun( )
				
				except Exception as e:
					try:
						conn.rollback( )
					except:
						pass
					st.error( f'Import failed — transaction rolled back.\n\n{e}' )
		
		# ------------------------------------------------------------------------------
		# BROWSE TAB
		# ------------------------------------------------------------------------------
		with tabs[ 1 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='table_name' )
				df = read_table( table )
				st.dataframe( df, use_container_width=True )
			else:
				st.info( 'No tables available.' )
		
		# ------------------------------------------------------------------------------
		# CRUD (Schema-Aware)
		# ------------------------------------------------------------------------------
		with tabs[ 2 ]:
			tables = list_tables( )
			if not tables:
				st.info( 'No tables available.' )
			else:
				table = st.selectbox( 'Select Table', tables, key='crud_table' )
				df = read_table( table )
				schema = create_schema( table )
				
				# Build type map
				type_map = { col[ 1 ]: col[ 2 ].upper( ) for col in schema if col[ 1 ] != 'rowid' }
				
				# ------------------------------------------------------------------
				# INSERT
				# ------------------------------------------------------------------
				st.subheader( 'Insert Row' )
				insert_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						insert_data[
							column ] = st.number_input( column, step=1, key=f'ins_{column}' )
					
					elif 'REAL' in col_type:
						insert_data[
							column ] = st.number_input( column, format='%.6f', key=f'ins_{column}' )
					
					elif 'BOOL' in col_type:
						insert_data[
							column ] = 1 if st.checkbox( column, key=f'ins_{column}' ) else 0
					
					else:
						insert_data[ column ] = st.text_input( column, key=f'ins_{column}' )
				
				if st.button( 'Insert Row' ):
					cols = list( insert_data.keys( ) )
					placeholders = ', '.join( [ '?' ] * len( cols ) )
					stmt = f'INSERT INTO "{table}" ({", ".join( cols )}) VALUES ({placeholders});'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( insert_data.values( ) ) )
						conn.commit( )
					
					st.success( 'Row inserted.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# UPDATE
				# ------------------------------------------------------------------
				st.subheader( 'Update Row' )
				rowid = st.number_input( 'Row ID', min_value=1, step=1 )
				update_data = { }
				for column, col_type in type_map.items( ):
					if 'INT' in col_type:
						val = st.number_input( column, step=1, key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'REAL' in col_type:
						val = st.number_input( column, format='%.6f', key=f'upd_{column}' )
						update_data[ column ] = val
					
					elif 'BOOL' in col_type:
						val = 1 if st.checkbox( column, key=f'upd_{column}' ) else 0
						update_data[ column ] = val
					
					else:
						val = st.text_input( column, key=f"upd_{column}" )
						update_data[ column ] = val
				
				if st.button( 'Update Row' ):
					set_clause = ', '.join( [ f'{c}=?' for c in update_data ] )
					stmt = f'UPDATE {table} SET {set_clause} WHERE rowid=?;'
					
					with create_connection( ) as conn:
						conn.execute( stmt, list( update_data.values( ) ) + [ rowid ] )
						conn.commit( )
					
					st.success( 'Row updated.' )
					st.rerun( )
				
				# ------------------------------------------------------------------
				# DELETE
				# ------------------------------------------------------------------
				st.subheader( 'Delete Row' )
				delete_id = st.number_input( 'Row ID to Delete', min_value=1, step=1 )
				if st.button( 'Delete Row' ):
					with create_connection( ) as conn:
						conn.execute( f'DELETE FROM {table} WHERE rowid=?;', (delete_id,) )
						conn.commit( )
					
					st.success( 'Row deleted.' )
					st.rerun( )
		
		# ------------------------------------------------------------------------------
		# EXPLORE
		# ------------------------------------------------------------------------------
		with tabs[ 3 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='explore_table' )
				page_size = st.slider( 'Rows per page', 10, 500, 50 )
				page = st.number_input( 'Page', min_value=1, step=1 )
				offset = (page - 1) * page_size
				df_page = read_table( table, page_size, offset )
				st.dataframe( df_page, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# FILTER
		# ------------------------------------------------------------------------------
		with tabs[ 4 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='filter_table' )
				df = read_table( table )
				column = st.selectbox( 'Column', df.columns )
				value = st.text_input( 'Contains' )
				if value:
					df = df[ df[ column ].astype( str ).str.contains( value ) ]
				st.dataframe( df, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# AGGREGATE
		# ------------------------------------------------------------------------------
		with tabs[ 5 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='agg_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols )
					agg = st.selectbox( 'Function', [ 'SUM', 'AVG', 'COUNT' ] )
					if agg == 'SUM':
						st.metric( 'Result', df[ col ].sum( ) )
					elif agg == 'AVG':
						st.metric( 'Result', df[ col ].mean( ) )
					elif agg == 'COUNT':
						st.metric( 'Result', df[ col ].count( ) )
		
		# ------------------------------------------------------------------------------
		# VISUALIZE
		# ------------------------------------------------------------------------------
		with tabs[ 6 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='viz_table' )
				df = read_table( table )
				numeric_cols = df.select_dtypes( include=[ 'number' ] ).columns.tolist( )
				if numeric_cols:
					col = st.selectbox( 'Column', numeric_cols, key='viz_column' )
					fig = px.histogram( df, x=col )
					st.plotly_chart( fig, use_container_width=True )
		
		# ------------------------------------------------------------------------------
		# ADMIN
		# ------------------------------------------------------------------------------
		with tabs[ 7 ]:
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Table', tables, key='admin_table' )
			
			st.divider( )
			
			st.subheader( 'Data Profiling' )
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='profile_table' )
				if st.button( 'Generate Profile' ):
					profile_df = create_profile_table( table )
					st.dataframe( profile_df, use_container_width=True )
			
			st.subheader( 'Drop Table' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table to Drop', tables, key='admin_drop_table' )
				
				# Initialize confirmation state
				if 'dm_confirm_drop' not in st.session_state:
					st.session_state.dm_confirm_drop = False
				
				# Step 1: Initial Drop click
				if st.button( 'Drop Table', key='admin_drop_button' ):
					st.session_state.dm_confirm_drop = True
				
				# Step 2: Confirmation UI
				if st.session_state.dm_confirm_drop:
					st.warning( f'You are about to permanently delete table {table}. '
					            'This action cannot be undone.' )
					
					col1, col2 = st.columns( 2 )
					
					if col1.button( 'Confirm Drop', key='admin_confirm_drop' ):
						try:
							drop_table( table )
							st.success( f'Table {table} dropped successfully.' )
						except Exception as e:
							st.error( f'Drop failed: {e}' )
						
						st.session_state.dm_confirm_drop = False
						st.rerun( )
					
					if col2.button( 'Cancel', key='admin_cancel_drop' ):
						st.session_state.dm_confirm_drop = False
						st.rerun( )
				
				df = read_table( table )
				col = st.selectbox( 'Create Index On', df.columns )
				
				if st.button( 'Create Index' ):
					create_index( table, col )
					st.success( 'Index created.' )
			
			st.divider( )
			
			st.subheader( 'Create Custom Table' )
			new_table_name = st.text_input( 'Table Name' )
			column_count = st.number_input( 'Number of Columns', min_value=1, max_value=20, value=1 )
			columns = [ ]
			for i in range( column_count ):
				st.markdown( f'### Column {i + 1}' )
				col_name = st.text_input( 'Column Name', key=f'col_name_{i}' )
				col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ],
					key=f'col_type_{i}' )
				
				not_null = st.checkbox( 'NOT NULL', key=f'not_null_{i}' )
				primary_key = st.checkbox( 'PRIMARY KEY', key=f'pk_{i}' )
				auto_inc = st.checkbox( 'AUTOINCREMENT (INTEGER only)', key=f'ai_{i}' )
				
				columns.append( {
						'name': col_name,
						'type': col_type,
						'not_null': not_null,
						'primary_key': primary_key,
						'auto_increment': auto_inc } )
			
			if st.button( 'Create Table' ):
				try:
					create_custom_table( new_table_name, columns )
					st.success( 'Table created successfully.' )
					st.rerun( )
				
				except Exception as e:
					st.error( f'Error: {e}' )
			
			st.divider( )
			st.subheader( 'Schema Viewer' )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='schema_view_table' )
				
				# Column schema
				schema = create_schema( table )
				schema_df = pd.DataFrame(
					schema,
					columns=[ 'cid', 'name', 'type', 'notnull', 'default', 'pk' ] )
				
				st.markdown( "### Columns" )
				st.dataframe( schema_df, use_container_width=True )
				
				# Row count
				with create_connection( ) as conn:
					count = conn.execute(
						f'SELECT COUNT(*) FROM "{table}"'
					).fetchone( )[ 0 ]
				
				st.metric( "Row Count", f"{count:,}" )
				
				# Indexes
				indexes = get_indexes( table )
				if indexes:
					idx_df = pd.DataFrame(
						indexes,
						columns=[ 'seq', 'name', 'unique', 'origin', 'partial' ]
					)
					st.markdown( "### Indexes" )
					st.dataframe( idx_df, use_container_width=True )
				else:
					st.info( "No indexes defined." )
			
			st.divider( )
			st.subheader( "ALTER TABLE Operations" )
			
			tables = list_tables( )
			if tables:
				table = st.selectbox( 'Select Table', tables, key='alter_table_select' )
				operation = st.selectbox( 'Operation',
					[ 'Add Column', 'Rename Column', 'Rename Table', 'Drop Column' ] )
				
				if operation == 'Add Column':
					new_col = st.text_input( 'Column Name' )
					col_type = st.selectbox( 'Column Type', [ 'INTEGER', 'REAL', 'TEXT' ] )
					
					if st.button( 'Add Column' ):
						add_column( table, new_col, col_type )
						st.success( 'Column added.' )
						st.rerun( )
				
				elif operation == 'Rename Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					old_col = st.selectbox( 'Column to Rename', col_names )
					new_col = st.text_input( 'New Column Name' )
					
					if st.button( 'Rename Column' ):
						rename_column( table, old_col, new_col )
						st.success( 'Column renamed.' )
						st.rerun( )
				
				elif operation == 'Rename Table':
					new_name = st.text_input( 'New Table Name' )
					
					if st.button( 'Rename Table' ):
						rename_table( table, new_name )
						st.success( 'Table renamed.' )
						st.rerun( )
				
				elif operation == 'Drop Column':
					schema = create_schema( table )
					col_names = [ col[ 1 ] for col in schema ]
					
					drop_col = st.selectbox( 'Column to Drop', col_names )
					
					if st.button( 'Drop Column' ):
						drop_column( table, drop_col )
						st.success( 'Column dropped.' )
						st.rerun( )
		
		# ------------------------------------------------------------------------------
		# SQL
		# ------------------------------------------------------------------------------
		with tabs[ 8 ]:
			st.subheader( 'SQL Console' )
			query = st.text_area( 'Enter SQL Query' )
			if st.button( 'Run Query' ):
				if not is_safe_query( query ):
					st.error( 'Query blocked: Only read-only SELECT statements are allowed.' )
				else:
					try:
						start_time = time.perf_counter( )
						with create_connection( ) as conn:
							result = pd.read_sql_query( query, conn )
						
						end_time = time.perf_counter( )
						elapsed = end_time - start_time
						
						# ----------------------------------------------------------
						# Display Results
						# ----------------------------------------------------------
						st.dataframe( result, use_container_width=True )
						row_count = len( result )
						
						# ----------------------------------------------------------
						# Execution Metrics
						# ----------------------------------------------------------
						col1, col2 = st.columns( 2 )
						col1.metric( 'Rows Returned', f'{row_count:,}' )
						col2.metric( 'Execution Time (seconds)', f'{elapsed:.6f}' )
						
						# Optional slow query warning
						if elapsed > 2.0:
							st.warning( 'Slow query detected (> 2 seconds). Consider indexing.' )
						
						# ----------------------------------------------------------
						# Download
						# ----------------------------------------------------------
						if not result.empty:
							csv = result.to_csv( index=False ).encode( 'utf-8' )
							st.download_button( 'Download CSV', csv,
								'query_results.csv', 'text/csv' )
					
					except Exception as e:
						st.error( f'Execution failed: {e}' )

# ==============================================================================
# FOOTER — SECTION
# ==============================================================================
st.markdown(
	"""
	<style>
	.block-container {
		padding-bottom: 3rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

# ---- Fixed Container
st.markdown(
	"""
	<style>
	.boo-status-bar {
		position: fixed;
		bottom: 0;
		left: 0;
		width: 100%;
		background-color: rgba(17, 17, 17, 0.95);
		border-top: 1px solid #2a2a2a;
		padding: 10px 16px;
		font-size: 0.80rem;
		color: #35618c;
		z-index: 1000;
	}
	.boo-status-inner {
		display: flex;
		justify-content: space-between;
		align-items: center;
		max-width: 100%;
	}
	</style>
	""", unsafe_allow_html=True,)

# ======================================================================================
# FOOTER RENDERING
# ======================================================================================

right_parts: List[ str ] = [ ]
model = 'Bro'

mode_value = mode if mode is not None else st.session_state.get( 'mode' )
if mode_value:
	right_parts.append( f'Mode: {mode_value}' )

temperature = st.session_state.get( 'temperature' )
top_p = st.session_state.get( 'top_percent' )
top_k = st.session_state.get( 'top_k' )
frequency = st.session_state.get( 'frequency_penalty' )
presense = st.session_state.get( 'presense_penalty' )
repeat_penalty = st.session_state.get( 'repeat_penalty' )
max_tokens = st.session_state.get( 'max_tokens' )
context_window = st.session_state.get( 'context_window' )
cpu_threads = st.session_state.get( 'cpu_threads' )
repeat_window = st.session_state.get( 'repeat_window' )
use_semantic = st.session_state.get( 'use_semantic' )
basic_docs = st.session_state.get( 'basic_docs' )

# ------------------------------------------------------------------
# Parameter summary (show 0 values; suppress only when None)
# ------------------------------------------------------------------
if temperature is not None:
	right_parts.append( f'Temp: {float( temperature ):0.2f}' )

if top_p is not None:
	right_parts.append( f'Top-P: {float( top_p ):0.2f}' )

if top_k is not None:
	right_parts.append( f'Top-K: {int( top_k )}' )

if frequency is not None:
	right_parts.append( f'Freq: {float( frequency ):0.2f}' )

if presense is not None:
	right_parts.append( f'Presence: {float( presense ):0.2f}' )

if repeat_penalty is not None:
	right_parts.append( f'Repeat: {float( repeat_penalty ):0.2f}' )

if repeat_window is not None:
	right_parts.append( f'Repeat Window: {int( repeat_window )}' )

if max_tokens is not None:
	right_parts.append( f'Max Tokens: {int( max_tokens )}' )

if context_window is not None:
	right_parts.append( f'Context: {int( context_window )}' )

if cpu_threads is not None:
	right_parts.append( f'Threads: {int( cpu_threads )}' )

# ------------------------------------------------------------------
# Context flags (optional but useful)
# ------------------------------------------------------------------
if use_semantic is not None:
	right_parts.append( f'Semantic: {"On" if use_semantic else "Off"}' )

if isinstance( basic_docs, list ):
	right_parts.append( f'Docs: {len( basic_docs )}' )

right_text = ' ◽ '.join( right_parts ) if right_parts else '—'

# ---- Rendering Method
st.markdown(
	f"""
    <div class="boo-status-bar">
        <div class="boo-status-inner">
            <span>{model}</span>
            <span>{right_text}</span>
        </div>
    </div>
    """,
	unsafe_allow_html=True, )