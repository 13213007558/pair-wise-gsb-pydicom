.. _api_signature:

Digital signatures (:mod:`pydicom.signature`)
=============================================

.. currentmodule:: pydicom.signature

Signature-aware edit planning. Stage changes to a dataset, inspect which
digital signatures they would invalidate, and then reject the edit or
remove the affected signatures before applying the changes.

.. autosummary::
   :toctree: generated/

   SignatureEditPlan
   InvalidatedSignature
   SignatureInvalidationError
