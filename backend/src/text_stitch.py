def stitch_overlapping_transcripts(transcripts: list, overlap_word_threshold: int = 5) -> str:
    """
    Stitches a list of sequential transcripts that have overlapping audio windows.
    Detects and eliminates duplicate phrase loops at the boundaries.
    
    Args:
        transcripts (list): List of string text outputs from each chunk sequence.
        overlap_word_threshold (int): Max number of words to check backwards for a match.
    """
    if not transcripts:
        return ""
    
    # Filter empty items and split transcripts into lists of words
    chunks = [t.strip().split() for t in transcripts if t.strip()]
    if not chunks:
        return ""
        
    stitched_words = chunks[0]
    
    # Iterate through the remaining text chunks sequentially
    for next_chunk in chunks[1:]:
        if not next_chunk:
            continue
            
        max_overlap = 0
        # Check window bounds up to the threshold or length of current sequence limits
        search_limit = min(len(stitched_words), len(next_chunk), overlap_word_threshold)
        
        # Slide backward along the tail end of our stitched corpus string
        for i in range(1, search_limit + 1):
            # Grab the last 'i' words of our current total text stream
            tail = stitched_words[-i:]
            # Grab the first 'i' words of the incoming chunk text string
            head = next_chunk[:i]
            
            # If the word lists match identically, log the current overlap window length
            if tail == head:
                max_overlap = i
                
        # Append only the unique, non-overlapping tail words from the new chunk
        stitched_words.extend(next_chunk[max_overlap:])
        
    return " ".join(stitched_words)