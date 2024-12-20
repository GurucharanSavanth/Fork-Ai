from sqlalchemy import create_engine, Column, String, Integer, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
import json

Base = declarative_base()

class Citation(Base):
    """Database model for academic citations."""
    __tablename__ = 'citations'

    id = Column(Integer, primary_key=True)
    cite_key = Column(String)  # Add cite_key column
    title = Column(String, nullable=False)
    authors = Column(String, nullable=False)
    doi = Column(String, unique=True, nullable=False)
    full_text_hash = Column(String)
    verified = Column(Boolean, default=False)

    def __repr__(self):
        return f"<Citation(title='{self.title}', doi='{self.doi}')>"

class CitationDB:
    """Database manager for citations."""
    def __init__(self, db_url='sqlite:///citations.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_citation(self, cite_key, title, authors, doi, full_text_hash=None, verified=False):
        """Add a citation to the database."""
        session = self.Session()
        try:
            # Convert authors list to JSON string for storage
            if isinstance(authors, list):
                authors = json.dumps(authors)

            citation = Citation(
                cite_key=cite_key,
                title=title,
                authors=authors,
                doi=doi,
                full_text_hash=full_text_hash,
                verified=verified
            )
            session.add(citation)
            session.commit()
            session.refresh(citation)  # Refresh to ensure all attributes are loaded
            return citation
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def get_citation(self, identifier):
        """Get a citation by its DOI or cite_key."""
        session = self.Session()
        try:
            # Try to find by DOI first
            citation = session.query(Citation).filter(Citation.doi == identifier).first()
            if citation:
                session.refresh(citation)
                return citation

            # If not found by DOI, try cite_key
            citation = session.query(Citation).filter(Citation.cite_key == identifier).first()
            if citation:
                session.refresh(citation)
            return citation
        finally:
            session.close()

    def verify_citation(self, doi):
        """Mark a citation as verified."""
        session = self.Session()
        try:
            citation = session.query(Citation).filter(Citation.doi == doi).first()
            if citation:
                citation.verified = True
                session.commit()
                return True
            return False
        finally:
            session.close()

def init_db(db_url='sqlite:///citations.db'):
    """Initialize the database with the schema."""
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()
