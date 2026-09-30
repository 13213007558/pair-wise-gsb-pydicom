.. _working_with_overlay_data:
.. title:: Working with Overlay Data

Working with Overlay Data
=========================

.. currentmodule:: pydicom

:dcm:`Overlays<part03/sect_C.9.2.html>` in DICOM are present in what's called
a :dcm:`Repeating Group<part05/sect_7.6.html>`, where the group number of the
element tags are defined over a range rather than a specific value. For
example, the tag's group number for (60xx,3000) *Overlay Data* may be (in hex)
``6000``, ``6002``, or any even value up to ``601E``. This allows a dataset to
include multiple overlays, where the related elements for each overlay use the
same group number. Because of this, the only way to access a particular
element from an overlay is to use the ``Dataset[group, elem]`` method::

  >>> from pydicom import examples
  >>> ds = examples.overlay
  >>> elem = ds[0x6000, 0x3000]  # returns a DataElement
  >>> print(elem)
  (6000, 3000) Overlay Data                        OW: Array of 29282 elements


*pydicom* tends to be "lazy" in interpreting DICOM data. For example, by default
it doesn't do anything with overlay data except read in the raw bytes::

  >>> elem.value # doctest: +ELLIPSIS
  b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00...

``Dataset.overlay_array()``
---------------------------

.. warning::

   :meth:`Dataset.overlay_array()<pydicom.dataset.Dataset.overlay_array>`
   requires `NumPy <https://numpy.org/>`_.

The *Overlay Data* element contains the raw bytes exactly as found in the file
as bit-packed data. To unpack and get an overlay in a more useful form you
can use the :meth:`~pydicom.dataset.Dataset.overlay_array` method to return a
:class:`numpy.ndarray`. To use it you only need to pass the group number of the
overlay elements you're interested in::

  >>> arr = ds.overlay_array(0x6000) # doctest: +NORMALIZE_WHITESPACE
  >>> arr
  array([[ 0, 0, 0, ...,  0,  0,  0],
         [ 0, 0, 0, ...,  0,  0,  0],
         [ 0, 0, 0, ...,  0,  0,  0],
         ...,
         [ 0, 0, 0, ...,  0,  0,  0],
         [ 0, 0, 0, ...,  0,  0,  0],
         [ 0, 0, 0, ...,  0,  0,  0],], dtype=uint8)
  >>> arr.shape
  (484, 484)

One thing to remember when dealing with *Overlay Data* is that the top left
of the overlay doesn't necessarily have to line up with the top left of the
related *Pixel Data*. The actual offset between them can be determined from
(60xx,0050) *Overlay Origin*, where a value of ``[1, 1]`` indicates that
the top left pixels are aligned and a value of ``[0, 0]`` indicates that the
overlay pixels start 1 row above and 1 row to the left of the image pixels.

NumPy can be used to modify the mask, and the modified mask can be written
back to the dataset with
:meth:`Dataset.set_overlay_array()
<pydicom.dataset.Dataset.set_overlay_array>`, which bit-packs the array,
handles partial-byte rows and odd-length padding and updates the related
elements (*Overlay Rows*, *Overlay Columns*, *Number of Frames in Overlay*,
*Overlay Origin*, *Overlay Bits Allocated* and *Overlay Bit Position*)::

  # Add a line
  arr[10, :] = 1

  # Pack the array and update the overlay elements
  ds.set_overlay_array(0x6000, arr)
  ds.save_as("temp.dcm")

By default the mask is written to the (60xx,3000) *Overlay Data* element.
The retired convention of embedding overlays in a high bit plane of the
*Pixel Data* is also supported when explicitly requested, provided the pixel
data is uncompressed and the bit plane doesn't overlap *Bits Stored*::

  # Embed the overlay at bit position 15 (Bits Stored 12)
  ds.set_overlay_array(0x6000, arr, bit_plane=15)

For embedded overlays the mask is mapped onto the pixels using the
``origin`` and ``image_frame_origin`` keyword arguments. Parts of the mask
that fall outside the pixel matrix raise an exception unless ``crop=True``
is given, in which case only the overlapping region is written.
