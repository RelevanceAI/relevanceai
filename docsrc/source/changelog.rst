Changelog
=================

Here you will find a list of changes for each package update.

v0.30.1
--------
**BREAKING CHANGES**

- `insert_csv` can now no longer be done via client. Needs to be done via `df.insert_csv`. Remapped to `client._insert_csv`
to avoid conflicting type errors.

Non-breaking changes:

- Fixed incorrect reference in `update_documents`



v0.30.0
---------

**BREAKING CHANGES**

- Renamed all `docs` references to `documents`
- Renamed all `cluster_alias` references to `alias`
- Changed functionality in CentroidClusterBase
- Renamed chunk_size to chunskize in get_all_documents
- Renamed `retrieve_chunk_size` to `retrieve_chunksize` in `df.apply` and `df.bulk_apply`
- Schema is now a property and not a method!
- `get_centroid_documents` now no longer takes a field
- Removal of any mention of `centroid_vector_` as those should now be replaced with the 
actual vector field name the centroids are derived from

Non-breaking changes:  

- Added `head` to Series object
- Add CentroidClustererbase and CentroidClusterBase classes to inherit from
- Deprecated KMeansClusterer in documentation and functionality
- Add fix for clusterer for missing vectors in documents by forcing filters
- Support for multi-region base URL based on frontend parsing
- Added AutoAPI to gitignore as we no longer want to measure that
- Add tighter sklearn integration
- Add CentroidClusterBase
- Clean up references around Clusterbase, Clusterer, Dataset
- Add reference to Client object
- Hotfix .sample()
- Update the Base Ingest URL to gateway and set to appropriate default
- Added support for base url token
- Removed QC from references
- Add integration reference
- Fixed centroid insertion for Dataset
- Refactor of tests based
- Add clustering test around clustering
- Separation of references to clean up clustering and sidebar menu navigation
- Fix reference examples

AUTO-GENERATED RELEASE NOTES:  

- Update README.md by @JackyKoh in https://github.com/RelevanceAI/RelevanceAI/pull/314
- Feature/refactor docsrc by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/315
- hotfix sample by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/316
- add installation suggestion by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/317
- Renaming docs to documents and cluster_alis to alias by @charyeezy in https://github.com/RelevanceAI/RelevanceAI/pull/308
- added column value to df.info by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/321
- update ingest to gateway by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/318
- Feature/remove qc by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/322
- Feature/separate centroid cluster bases by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/323
- Feature/fix series object by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/324
- Renaming datasets by @charyeezy in https://github.com/RelevanceAI/RelevanceAI/pull/320
- add integration RST and code improvements by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/326
- added df.filter to dataset api by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/319
- Reference Quality check by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/325
- Feature/fix docsrc 2 by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/328
- Fixing notebook test by @charyeezy in https://github.com/RelevanceAI/RelevanceAI/pull/327
- Feature/fix example custom cluster model by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/329
- fixed centroids by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/330
- add core by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/331
- Update documentation on kmeans cluster model  by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/332
- Feature/fix references 3 by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/334
- added kmeans integration by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/333


v0.29.1
---------

- Moved get_all_documents in BatchAPIClient to _get_all_documents to resolve typing error
- Include Client, Fix Clusterer, ClusterBase, update Cluster References
- Add Write Documentation by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/311
- update clustering documentation and client documentation by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/312


v0.29.0
--------

- Added value_counts method to Dataset API by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/272
- Added to_dict for pandas dataset api by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/293
- Feature/add clusterer object by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/306
- Feature/fix references docs by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/302
- Feature/edit docs by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/309

v0.28.2
--------

- Update RELEASES.md by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/287
- Feature/make conda installable by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/288
- Concatentate Numeric Features into Vector by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/289
- from_csv and to_csv - Dataset API by @jtwinrelevanceai in https://github.com/RelevanceAI/RelevanceAI/pull/281
- Fixing hybrid search field by @charyeezy in https://github.com/RelevanceAI/RelevanceAI/pull/285
- created mean method for GroupBy and corresponding test by @ofrighil in https://github.com/RelevanceAI/RelevanceAI/pull/291
- Add link by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/299
- Feature/pinning notebook version to 0.27.0 in notebook tests by @charyeezy in https://github.com/RelevanceAI/RelevanceAI/pull/301
- Update centroid documents and restructure docs  by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/300
- make alias required by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/296
- @ofrighil made their first contribution in https://github.com/RelevanceAI/RelevanceAI/pull/291


v0.28.1
--------

- removed clustering results from get_realestate_dataset by @ChakavehSaedi in https://github.com/RelevanceAI/RelevanceAI/pull/277
- add option to print no dashboard by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/278
- move to node implementation for listing furthest by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/279
- add output field to apply by @boba-and-beer in https://github.com/RelevanceAI/RelevanceAI/pull/282
- Add releases workflow markdown and diagram
- Fix clustering tests

v0.28.0
--------

- *Breaking Change*️ Change pull_update_push to use dataset ID
- Added centroid distance evaluation
- Added JSONShower to df.head() so previewing images is now possible
- Refactor Pandas Dataset API to use BatchAPIClient
- Modularise testing infrastructure to use separate datasets
- Add aggregation, groupby pandas API support
- Added GroupBy, Series class for Datasets
- Added datasets.info()
- Added documentation testing
- Added df.apply()
- Added additional functionality for sampling etc.
- Fixed documentation for Datasets API
- Add new monitoring health test for chunk data structure
- Add fix for csv reading for _chunk_ to be parsed as actual Python objects
and not strings

v0.27.0
--------

- Fixed datasets.documents.update_where so it runs
- Added more tests around multivector search
- Added Pandas-like Dataset Class for interacting with SDK (Alpha)
- Added datasets.cluster.centroids.list_furthest_from_centers and datasets.cluster.centroids.list_closest_to_centers
- Folder Refactor

v0.26.6
--------

- Fix missing import in plotting since internalising plots
- Add support for vector labels
- Remove background axes from plot

v0.26.5
---------

- Fix incorrect URL being submitted to frontend

v0.26.4
---------

- Fix string parsing issue for endpoints and dashboards

v0.26.3
---------

- Cluster labels are now lower case 
- Bug fix on centroids furthest from center
- Changed error message 
- Fixed Dodgy string parsing
- Fixed bug with kmeans_cluster 1 liner by supporting getting multiple centers

v0.26.2
---------

- Add CSV insertion 
- Make JSON encoder utility class for easier customisation
- Added smarter parsing of CSV

v0.26.1
---------

- Bug fixes

v0.26.0
---------

- Added JSON serialization and consequent test updates
- Bug fix to cluster metrics
- Minor fix to tests
