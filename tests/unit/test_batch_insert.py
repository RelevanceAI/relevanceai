"""Testing code for batch inserting
"""

class TestInsert:
    """Testing the insert functionalities
    """
    def test_batch_insert(self, simple_docs, test_dataset_id, 
        test_client):
        """Batch insert
        """
        results = test_client.insert_documents(
            test_dataset_id, simple_docs)
        assert len(results['failed_documents']) == 0
    
class TestPullUpdatePush:
    """Testing Pull Update Push
    """
    def test_pull_update_push_simple(self, test_dataset_id, 
        test_client):
        """Simple test for pull update push
        """
        
        # Sample function
        def do_nothing(docs):
            return docs
        
        results = test_client.pull_update_push(test_dataset_id, do_nothing)
        assert len(results['failed_documents']) == 0

class CleanUp:
    def test_clean_up(self, test_client, test_dataset_id):
        assert test_client.datasets.delete(test_dataset_id)
