function [y] = circular_mean(m_sin,m_cos)
    if and(m_sin>0 , m_cos>0)
        y = atan(m_sin./m_cos);
    elseif and(m_sin<0 , m_cos>0)
        y = atan(m_sin./m_cos);
    elseif and(m_sin<0 , m_cos<0)
        y = atan(m_sin./m_cos)-pi;
    elseif and(m_sin>0 , m_cos<0)
        y = atan(m_sin./m_cos)+pi;
    end
    
end