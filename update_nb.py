import json
import os

filepath = 'LLM_Engineering/query_construction.ipynb'
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

sql_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Text to SQL\n", "\n", "Translating natural language to SQL queries using LangChain's SQLDatabaseChain."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from langchain_community.utilities import SQLDatabase\n",
            "from langchain_experimental.sql import SQLDatabaseChain\n",
            "from sqlalchemy import create_engine, Column, Integer, String, Float\n",
            "from sqlalchemy.orm import declarative_base, sessionmaker\n",
            "\n",
            "# 1. Setup an in-memory SQLite database for the example\n",
            "engine = create_engine('sqlite:///:memory:')\n",
            "Base = declarative_base()\n",
            "\n",
            "class Movie(Base):\n",
            "    __tablename__ = 'movies'\n",
            "    id = Column(Integer, primary_key=True)\n",
            "    title = Column(String)\n",
            "    genre = Column(String)\n",
            "    year = Column(Integer)\n",
            "    director = Column(String)\n",
            "    rating = Column(Float)\n",
            "\n",
            "Base.metadata.create_all(engine)\n",
            "Session = sessionmaker(bind=engine)\n",
            "session = Session()\n",
            "\n",
            "# Add some sample data\n",
            "movies = [\n",
            "    Movie(title='The Shawshank Redemption', genre='drama', year=1994, director='Frank Darabont', rating=9.3),\n",
            "    Movie(title='Pulp Fiction', genre='thriller', year=1994, director='Quentin Tarantino', rating=8.9),\n",
            "    Movie(title='The Dark Knight', genre='action', year=2008, director='Christopher Nolan', rating=9.0),\n",
            "    Movie(title='Greta', genre='thriller', year=2018, director='Neil Jordan', rating=6.0),\n",
            "    Movie(title='Barbie', genre='comedy', year=2023, director='Greta Gerwig', rating=7.0),\n",
            "]\n",
            "session.add_all(movies)\n",
            "session.commit()\n",
            "\n",
            "db = SQLDatabase(engine)\n",
            "db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)\n",
            "\n",
            "def text_to_sql_demo(query):\n",
            "    print(f'\\nUser Request: {query}')\n",
            "    try:\n",
            "        result = db_chain.run(query)\n",
            "        print(f'Result: {result}')\n",
            "    except Exception as e:\n",
            "        print(f'Error: {e}')\n",
            "\n",
            "text_to_sql_demo('How many movies are there?')\n",
            "text_to_sql_demo('What is the highest rated movie from 1994?')\n",
            "text_to_sql_demo('List all comedy movies')"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Text to SQL + Semantic Search\n", "\n", "This setup combines structured SQL-like filtering with semantic content search, similar to a hybrid search approach."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Conceptual example of combining SQL (structured) and Vector (semantic)\n",
            "\n",
            "def hybrid_query(query):\n",
            "    print(f'\\nHybrid Request: {query}')\n",
            "    # 1. Structured part: Extract filters (like we did in the first section)\n",
            "    # 2. Semantic part: Content search query\n",
            "    \n",
            "    result = chain.invoke({'query': query})\n",
            "    \n",
            "    content_query = result.query.strip() or \"(none)\"\n",
            "    print(f'Content Query (Semantic): {content_query}')\n",
            "    print(f'Metadata Filter (Structured):')\n",
            "    from pprint import pprint\n",
            "    pprint(readable_filter(result.filter))\n",
            "\n",
            "hybrid_query('Find a movie about a bank heist that was released after 2010 with a rating above 8')"
        ]
    }
]

data['cells'].extend(sql_cells)

with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)
